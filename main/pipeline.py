"""
Pipeline orchestrator - ties the full translation workflow together.

Flow: Load image → Detect text (ML model) → OCR → Translate → Inpaint + Render → Save
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image as PILImage

from .config import Settings, settings
from .detector import TextRegion, detect_text_regions
from .ocr import get_ocr_engine
from .renderer import detect_text_color, inpaint_text_region, render_text_on_image
from .translator import TranslationConstraint, translate_texts

logger = logging.getLogger(__name__)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# WebP / JPEG quality targets that produce sizes close to professional scanlations.
_WEBP_QUALITY = 82
_WEBP_METHOD = 6  # slower but best compression
_JPEG_QUALITY = 85


@dataclass
class RenderTextUnit:
    """A renderable text unit, potentially split from a larger detected region."""

    region: TextRegion
    context_region: TextRegion | None
    source_text: str
    style_hint: str
    parent_region_index: int
    text_color: tuple[int, int, int] | None = None


def translate_page(
    image_path: str | Path,
    output_path: str | Path | None = None,
    config: Settings | None = None,
) -> Path:
    """
    Translate a single manga page end-to-end.

    Args:
        image_path: Path to the input manga page image.
        output_path: Path to save the translated image. Auto-generated if None.
        config: Settings override. Uses global settings if None.

    Returns:
        Path to the saved translated image.
    """
    cfg = config or settings
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    logger.info("=" * 60)
    logger.info("Processing: %s", image_path.name)
    logger.info("=" * 60)

    # 1. Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    logger.info("Image loaded: %dx%d", image.shape[1], image.shape[0])

    # 2. Detect text regions using ML model
    logger.info("Step 1/4: Detecting text regions (ML model)...")
    regions, text_mask = detect_text_regions(image)
    logger.info("Found %d text regions.", len(regions))

    if not regions:
        logger.warning("No text regions detected. Saving original image.")
        out = _resolve_output_path(image_path, output_path, cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        _save_image(out, image)
        return out

    # 3. OCR - extract text from each region
    logger.info("Step 2/4: Extracting text (OCR, lang=%s)...", cfg.source_lang)
    ocr_engine = get_ocr_engine(cfg.source_lang)
    units: list[RenderTextUnit] = []
    inpaint_region_indices: set[int] = set()

    # Pre-split all regions into OCR sub-regions
    all_ocr_tasks: list[tuple[int, int, TextRegion]] = []
    for region_idx, region in enumerate(regions):
        ocr_regions = _split_region_for_ocr(region)
        if len(ocr_regions) >= 2:
            logger.info(
                "  Region %d split into %d OCR sub-regions.",
                region_idx + 1,
                len(ocr_regions),
            )
        for sub_idx, ocr_region in enumerate(ocr_regions, 1):
            all_ocr_tasks.append((region_idx, sub_idx, ocr_region))

    # Run OCR — parallel (3 workers) for Modal, sequential for local CPU
    def _ocr_one(task: tuple[int, int, TextRegion]) -> tuple[int, int, TextRegion, str]:
        region_idx, sub_idx, ocr_region = task
        text = ocr_engine.extract_text(ocr_region.cropped, ocr_region.mask).strip()
        return region_idx, sub_idx, ocr_region, text

    if cfg.use_modal and len(all_ocr_tasks) > 1:
        from concurrent.futures import ThreadPoolExecutor

        max_workers = min(3, len(all_ocr_tasks))
        logger.info("  Running OCR in parallel (%d workers, %d sub-regions)...",
                     max_workers, len(all_ocr_tasks))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            ocr_results = list(pool.map(_ocr_one, all_ocr_tasks))
    else:
        ocr_results = [_ocr_one(t) for t in all_ocr_tasks]

    for region_idx, sub_idx, ocr_region, text in ocr_results:
        if not text:
            continue

        style_hint = _infer_unit_style(text, ocr_region.w, ocr_region.h)
        parent_region = regions[region_idx]
        units.append(
            RenderTextUnit(
                region=ocr_region,
                context_region=_select_render_context_region(
                    unit_region=ocr_region,
                    parent_region=parent_region,
                    style_hint=style_hint,
                ),
                source_text=text,
                style_hint=style_hint,
                parent_region_index=region_idx,
            )
        )
        inpaint_region_indices.add(region_idx)
        logger.info(
            "  Region %d.%d [%s]: '%s'",
            region_idx + 1,
            sub_idx,
            style_hint,
            text[:80],
        )

    if not units:
        logger.warning("No text extracted from any region. Saving original image.")
        out = _resolve_output_path(image_path, output_path, cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        _save_image(out, image)
        return out

    # 4. Translate
    logger.info("Step 3/4: Translating %d text segments...", len(units))
    source_texts = [unit.source_text for unit in units]
    constraints = [
        _build_translation_constraint(unit)
        for unit in units
    ]
    translated_texts = translate_texts(
        source_texts,
        source_lang=cfg.source_lang,
        model=cfg.translation_model,
        constraints=constraints,
    )
    # Light sanitization: only block CJK leakage, let renderer handle sizing
    translated_texts = [
        _sanitize_translated_text(text=text)
        for text in translated_texts
    ]
    renderable_unit_mask = [
        _is_renderable_translation(text) for text in translated_texts
    ]
    renderable_region_indices = {
        unit.parent_region_index
        for unit, can_render in zip(units, renderable_unit_mask)
        if can_render
    }

    for i, (src, tgt) in enumerate(zip(source_texts, translated_texts)):
        logger.info("  [%d] %s → %s", i + 1, src[:40], tgt[:60])
    skipped_units = len(renderable_unit_mask) - sum(renderable_unit_mask)
    if skipped_units > 0:
        logger.warning(
            "Skipping %d unit(s) with empty/non-renderable translated text.",
            skipped_units,
        )
    if inpaint_region_indices and not renderable_region_indices:
        logger.warning(
            "No renderable translations for this page; preserving original text regions."
        )

    # 5. Detect original text colours *before* inpainting erases them.
    logger.info("Step 4/5: Detecting original text colours...")
    for unit, can_render in zip(units, renderable_unit_mask):
        if not can_render:
            continue
        region = unit.region
        unit.text_color = detect_text_color(
            region_image=region.cropped,
            region_mask=region.mask,
        )
        if unit.text_color is not None:
            logger.info(
                "  Region at (%d,%d): detected colour RGB%s",
                region.x,
                region.y,
                unit.text_color,
            )

    # 6. Inpaint original text, then render translated text
    logger.info("Step 5/5: Inpainting and rendering translated text...")
    result = image.copy()

    # First pass: inpaint only regions that have at least one renderable translation.
    # This avoids blank bubbles/pages when translation output is empty or unusable.
    for region_idx in sorted(renderable_region_indices):
        region = regions[region_idx]
        result = inpaint_text_region(
            result, region.x, region.y, region.w, region.h, region.mask
        )

    # Second pass: render translated text onto the cleaned image
    for unit, translated, can_render in zip(units, translated_texts, renderable_unit_mask):
        if can_render:
            region = unit.context_region or unit.region
            render_mask = _project_region_mask_into_context(
                unit_region=unit.region,
                context_region=region,
            )
            result = render_text_on_image(
                result,
                translated,
                region.x,
                region.y,
                region.w,
                region.h,
                region_mask=render_mask,
                style_hint=unit.style_hint,
                text_color=unit.text_color,
            )

    # 6. Save output
    out = _resolve_output_path(image_path, output_path, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_image(out, result)
    logger.info("Saved translated image: %s", out)
    logger.info("=" * 60)

    return out


def translate_directory(
    input_dir: str | Path,
    config: Settings | None = None,
) -> list[Path]:
    """
    Translate all images in a directory.

    Args:
        input_dir: Directory containing manga page images.
        config: Settings override.

    Returns:
        List of paths to translated images.
    """
    cfg = config or settings
    input_dir = Path(input_dir)

    image_files = list_image_files(input_dir)

    if not image_files:
        logger.warning("No image files found in %s", input_dir)
        return []

    logger.info("Found %d images to translate in %s", len(image_files), input_dir)
    return translate_images(image_files, config=cfg)


def translate_images(
    image_files: list[str | Path],
    config: Settings | None = None,
) -> list[Path]:
    """
    Translate a list of image files.

    Processing mode:
    - Local OCR (`USE_MODAL=false`): batched parallel workers
      (overlaps I/O-bound translation API calls with CPU-bound
      detection/OCR for ~30-40% throughput improvement).
    - Modal OCR (`USE_MODAL=true`): batched parallel workers
    """
    cfg = config or settings
    files = [Path(f) for f in image_files]
    total = len(files)

    if total == 0:
        logger.warning("No input images provided.")
        return []

    results: list[Path] = []
    tasks = list(enumerate(files, start=1))

    def _translate_one(task: tuple[int, Path]) -> tuple[Path | None, Exception | None]:
        i, img_path = task
        logger.info("\n[%d/%d] Processing %s...", i, total, img_path.name)
        try:
            out = translate_page(img_path, config=cfg)
            return out, None
        except Exception as e:
            return None, e

    # Determine parallelism based on backend
    if cfg.use_modal:
        max_workers = min(cfg.modal_max_parallel_pages, total)
        label = "Modal"
    else:
        max_workers = min(cfg.local_max_parallel_pages, total)
        label = "local"

    if max_workers > 1 and total > 1:
        total_batches = (total + max_workers - 1) // max_workers
        logger.info(
            "Using batched parallel page processing (%s, %d workers per batch).",
            label,
            max_workers,
        )
        for batch_index, start in enumerate(range(0, total, max_workers), start=1):
            batch_tasks = tasks[start : start + max_workers]
            logger.info(
                "Starting batch %d/%d with %d page(s).",
                batch_index,
                total_batches,
                len(batch_tasks),
            )
            with ThreadPoolExecutor(max_workers=len(batch_tasks)) as pool:
                batch_results = list(pool.map(_translate_one, batch_tasks))

            for (_, img_path), (out, error) in zip(batch_tasks, batch_results):
                if out is not None:
                    results.append(out)
                elif error is not None:
                    logger.error("Failed to process %s: %s", img_path.name, error)
    else:
        for task in tasks:
            _, img_path = task
            out, error = _translate_one(task)
            if out is not None:
                results.append(out)
            elif error is not None:
                logger.error("Failed to process %s: %s", img_path.name, error)

    logger.info("\nDone! Translated %d/%d images.", len(results), total)
    return results


def is_image_file(path: str | Path) -> bool:
    """Return True if `path` is a supported image file."""
    p = Path(path)
    return p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS


def list_image_files(input_dir: str | Path) -> list[Path]:
    """List supported image files in a directory."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_path}")
    return sorted(f for f in input_path.iterdir() if is_image_file(f))


def _resolve_output_path(
    image_path: Path, output_path: str | Path | None, cfg: Settings
) -> Path:
    """Determine the output file path."""
    if output_path:
        return Path(output_path)
    out_dir = cfg.output_dir
    return out_dir / image_path.name


def _save_image(output_path: Path, image_bgr: np.ndarray) -> None:
    """Save an image with format-aware compression to avoid file-size bloat.

    For WebP and JPEG, uses PIL with tuned quality settings that produce sizes
    close to professional scanlation quality.  Other formats fall back to
    cv2.imwrite for simplicity.
    """
    ext = output_path.suffix.lower()
    if ext in (".webp", ".jpg", ".jpeg"):
        # cv2 stores BGR; PIL expects RGB
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)
        if ext == ".webp":
            pil_img.save(
                str(output_path),
                format="WEBP",
                quality=_WEBP_QUALITY,
                method=_WEBP_METHOD,
            )
        else:
            pil_img.save(
                str(output_path),
                format="JPEG",
                quality=_JPEG_QUALITY,
                optimize=True,
            )
    else:
        cv2.imwrite(str(output_path), image_bgr)


def _select_render_context_region(
    unit_region: TextRegion,
    parent_region: TextRegion,
    style_hint: str,
) -> TextRegion:
    """Choose a render context that preserves bubble geometry for dialogue."""
    if style_hint != "dialogue":
        return unit_region
    if (
        unit_region.x == parent_region.x
        and unit_region.y == parent_region.y
        and unit_region.w == parent_region.w
        and unit_region.h == parent_region.h
    ):
        return unit_region
    return parent_region


def _project_region_mask_into_context(
    unit_region: TextRegion,
    context_region: TextRegion,
) -> np.ndarray | None:
    """Paste a unit mask into the chosen context region for bubble anchoring."""
    if unit_region.mask is None:
        return None
    if unit_region.mask.shape[:2] != (unit_region.h, unit_region.w):
        return None
    if (
        unit_region.x == context_region.x
        and unit_region.y == context_region.y
        and unit_region.w == context_region.w
        and unit_region.h == context_region.h
    ):
        return unit_region.mask.copy()

    offset_x = unit_region.x - context_region.x
    offset_y = unit_region.y - context_region.y
    if offset_x < 0 or offset_y < 0:
        return None
    if offset_x + unit_region.w > context_region.w:
        return None
    if offset_y + unit_region.h > context_region.h:
        return None

    projected = np.zeros((context_region.h, context_region.w), dtype=np.uint8)
    projected[
        offset_y : offset_y + unit_region.h,
        offset_x : offset_x + unit_region.w,
    ] = unit_region.mask.astype(np.uint8)
    return projected


def _build_translation_constraint(unit: RenderTextUnit) -> TranslationConstraint:
    """Derive concise translation budgets from the region most likely to be rendered."""
    render_region = unit.context_region or unit.region
    area = max(1, int(render_region.w * render_region.h))
    aspect = render_region.h / float(max(1, render_region.w))

    if unit.style_hint == "sfx":
        max_chars = 12 if area < 12000 else 16
        if render_region.w >= 160 or render_region.h >= 220:
            max_chars = min(18, max_chars + 2)
        return TranslationConstraint(
            style="sfx",
            max_words=3,
            max_chars=max_chars,
        )

    max_words = 7
    max_chars = 36
    if aspect >= 1.75:
        max_words = 3 if render_region.w < 120 or area < 24000 else 4
        max_chars = 16 if render_region.w < 120 else 20
    elif aspect >= 1.35:
        max_words = 4 if area < 26000 else 5
        max_chars = 20 if area < 26000 else 24
    elif area < 18000:
        max_words = 5
        max_chars = 24
    elif area < 32000:
        max_words = 6
        max_chars = 30

    return TranslationConstraint(
        style="dialogue",
        max_words=max_words,
        max_chars=max_chars,
    )


def _split_region_for_ocr(region: TextRegion) -> list[TextRegion]:
    """
    Split a merged region into component-level OCR units.

    This reduces the chance that unrelated nearby text is OCR'd as one long line.
    """
    if region.mask is None or region.mask.shape[:2] != (region.h, region.w):
        return [region]

    if region.mask.max() > 1:
        _, binary = cv2.threshold(region.mask, 127, 255, cv2.THRESH_BINARY)
    else:
        binary = (region.mask > 0).astype(np.uint8) * 255

    if cv2.countNonZero(binary) < 24:
        return [region]

    # Remove tiny speckles without destroying text strokes.
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 6 or h < 6:
            continue
        boxes.append((x, y, x + w, y + h))

    if len(boxes) < 2:
        return [region]

    merge_gap = max(8, int(min(region.w, region.h) * 0.10))
    merged_boxes = _merge_nearby_boxes(boxes, gap=merge_gap)
    merged_boxes = _merge_aligned_ocr_boxes(
        merged_boxes,
        region_w=region.w,
        region_h=region.h,
    )
    merged_boxes = [
        b
        for b in merged_boxes
        if (b[2] - b[0]) >= 12
        and (b[3] - b[1]) >= 12
        and ((b[2] - b[0]) * (b[3] - b[1])) >= 140
    ]

    if len(merged_boxes) < 2:
        return [region]
    if len(merged_boxes) > 8:
        merged_boxes = _merge_nearby_boxes(
            merged_boxes,
            gap=max(12, int(min(region.w, region.h) * 0.20)),
        )
    if len(merged_boxes) < 2 or len(merged_boxes) > 8:
        return [region]

    pad = max(2, int(min(region.w, region.h) * 0.03))
    sorted_boxes = _sort_local_manga_order(merged_boxes, region_width=region.w)
    units: list[TextRegion] = []
    for x1, y1, x2, y2 in sorted_boxes:
        ux1 = max(0, x1 - pad)
        uy1 = max(0, y1 - pad)
        ux2 = min(region.w, x2 + pad)
        uy2 = min(region.h, y2 + pad)
        if ux2 <= ux1 or uy2 <= uy1:
            continue

        sub_mask = binary[uy1:uy2, ux1:ux2].copy()
        if cv2.countNonZero(sub_mask) < 18:
            continue
        sub_crop = region.cropped[uy1:uy2, ux1:ux2].copy()
        units.append(
            TextRegion(
                x=region.x + ux1,
                y=region.y + uy1,
                w=ux2 - ux1,
                h=uy2 - uy1,
                cropped=sub_crop,
                mask=sub_mask,
            )
        )

    return units if len(units) >= 2 else [region]


def _merge_aligned_ocr_boxes(
    boxes: list[tuple[int, int, int, int]],
    region_w: int,
    region_h: int,
) -> list[tuple[int, int, int, int]]:
    """Merge OCR boxes that are clearly one phrase split across tight columns/rows."""
    if len(boxes) < 2:
        return boxes

    x_gap_limit = max(12, int(region_w * 0.16))
    y_gap_limit = max(10, int(region_h * 0.08))

    def _should_merge(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> bool:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        aw = max(1, ax2 - ax1)
        ah = max(1, ay2 - ay1)
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)

        x_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
        y_overlap = max(0, min(ay2, by2) - max(ay1, by1))
        x_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
        y_gap = max(0, max(ay1, by1) - min(ay2, by2))

        x_overlap_ratio = x_overlap / float(max(1, min(aw, bw)))
        y_overlap_ratio = y_overlap / float(max(1, min(ah, bh)))

        same_column = x_overlap_ratio >= 0.65 and y_gap <= y_gap_limit
        side_by_side_columns = (
            y_overlap_ratio >= 0.55
            and x_gap <= min(x_gap_limit, int(max(aw, bw) * 0.45) + 8)
        )
        return same_column or side_by_side_columns

    merged = boxes.copy()
    changed = True
    while changed:
        changed = False
        next_boxes: list[tuple[int, int, int, int]] = []
        used = [False] * len(merged)
        for i, base in enumerate(merged):
            if used[i]:
                continue
            x1, y1, x2, y2 = base
            used[i] = True
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if not _should_merge((x1, y1, x2, y2), merged[j]):
                    continue
                ox1, oy1, ox2, oy2 = merged[j]
                x1 = min(x1, ox1)
                y1 = min(y1, oy1)
                x2 = max(x2, ox2)
                y2 = max(y2, oy2)
                used[j] = True
                changed = True
            next_boxes.append((x1, y1, x2, y2))
        merged = next_boxes

    return merged


def _sort_local_manga_order(
    boxes: list[tuple[int, int, int, int]],
    region_width: int,
) -> list[tuple[int, int, int, int]]:
    """Sort local boxes in rough manga order (right-to-left, top-to-bottom)."""
    if not boxes:
        return []

    columns = 3 if region_width >= 180 else 2
    col_w = max(1.0, region_width / float(columns))

    def sort_key(box: tuple[int, int, int, int]) -> tuple[int, int]:
        x1, y1, x2, y2 = box
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        col = int(cx / col_w)
        col = columns - 1 - col
        return (col, int(cy))

    return sorted(boxes, key=sort_key)


def _boxes_are_near(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
    y_gap = max(0, max(ay1, by1) - min(ay2, by2))
    return x_gap <= gap and y_gap <= gap


def _merge_nearby_boxes(
    boxes: list[tuple[int, int, int, int]],
    gap: int,
) -> list[tuple[int, int, int, int]]:
    """Merge overlapping/nearby boxes until stable."""
    if not boxes:
        return []

    merged = boxes.copy()
    changed = True
    while changed:
        changed = False
        used = [False] * len(merged)
        next_boxes: list[tuple[int, int, int, int]] = []
        for i, box in enumerate(merged):
            if used[i]:
                continue
            x1, y1, x2, y2 = box
            used[i] = True
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if _boxes_are_near((x1, y1, x2, y2), merged[j], gap=gap):
                    ox1, oy1, ox2, oy2 = merged[j]
                    x1 = min(x1, ox1)
                    y1 = min(y1, oy1)
                    x2 = max(x2, ox2)
                    y2 = max(y2, oy2)
                    used[j] = True
                    changed = True
            next_boxes.append((x1, y1, x2, y2))
        merged = next_boxes
    return merged


def _infer_unit_style(source_text: str, w: int, h: int) -> str:
    """Coarse style hint: SFX for short sounds, dialogue for everything else."""
    clean = "".join(source_text.split())
    if not clean:
        return "sfx"

    script_total, kana_count, kanji_count, punct_count = _script_profile(clean)
    char_count = len(clean)
    has_sentence_punct = any(c in clean for c in "。！？?!")
    dialogue_markers = (
        "して",
        "ない",
        "れる",
        "られる",
        "です",
        "ます",
        "たい",
        "という",
        "かも",
        "から",
        "まで",
        "よう",
        "する",
        "した",
    )
    has_dialogue_marker = any(marker in clean for marker in dialogue_markers)

    # Short text without sentence punctuation is likely SFX
    if char_count <= 3:
        return "sfx"
    if has_dialogue_marker and char_count >= 4:
        return "dialogue"
    if kanji_count >= 1 and kana_count >= 2 and char_count >= 4:
        return "dialogue"
    if char_count <= 6 and not has_sentence_punct and kanji_count <= 1:
        return "sfx"
    return "dialogue"




def _sanitize_translated_text(text: str) -> str:
    """Guard against untranslated CJK leakage before rendering."""
    clean = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    # First pass: replace common unsupported unicode chars with ASCII equivalents
    clean = _replace_unsupported_chars(clean)
    if not _contains_cjk(clean):
        # Still strip stray CJK punctuation / fullwidth chars even if no CJK text.
        stripped = _strip_cjk_chars(clean)
        return stripped if stripped else clean
    # Strip CJK characters but keep the English/Latin text.
    stripped = _strip_cjk_chars(clean)
    return stripped if stripped else "..."


def _replace_unsupported_chars(text: str) -> str:
    """Replace common unicode chars that manga fonts can't render with ASCII equivalents."""
    replacements = {
        # Smart quotes → ASCII
        "\u2018": "'", "\u2019": "'", "\u201A": "'",
        "\u201C": '"', "\u201D": '"', "\u201E": '"',
        # Dashes
        "\u2013": "-", "\u2014": "-", "\u2012": "-",
        # Ellipsis
        "\u2026": "...",
        # Geometric shapes (these cause □ tofu)
        "\u25A0": "", "\u25A1": "", "\u25AA": "", "\u25AB": "",
        "\u25B2": "", "\u25B3": "", "\u25B6": "", "\u25BC": "",
        "\u25CF": "", "\u25CB": "", "\u25FB": "", "\u25FC": "",
        "\u25FD": "", "\u25FE": "",
        # Misc symbols that cause tofu
        "\u2B1C": "", "\u2B1B": "",  # Large squares
        "\u2605": "*", "\u2606": "*",  # Stars
        "\u2022": "-",  # Bullet
        "\u00B7": ".",  # Middle dot
        # Arrows
        "\u2192": "->", "\u2190": "<-", "\u2191": "^", "\u2193": "v",
        # Musical notes
        "\u266A": "~", "\u266B": "~", "\u266C": "~",
        # Hearts/symbols that may not render
        "\u2665": "<3", "\u2764": "<3",
        # Tildes
        "\u301C": "~", "\uFF5E": "~",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _strip_cjk_chars(text: str) -> str:
    """Remove CJK characters, CJK punctuation, fullwidth forms, and unsupported symbols."""
    out = []
    for ch in text:
        code = ord(ch)
        if (
            0x2500 <= code <= 0x25FF  # Box Drawing + Geometric Shapes (e.g. □, ■, △)
            or 0x2600 <= code <= 0x26FF  # Miscellaneous Symbols
            or 0x2700 <= code <= 0x27BF  # Dingbats
            or 0x3000 <= code <= 0x303F   # CJK symbols and punctuation (「」、。etc)
            or 0x3040 <= code <= 0x30FF  # Hiragana / Katakana
            or 0x31F0 <= code <= 0x31FF  # Katakana phonetic extensions
            or 0x3400 <= code <= 0x4DBF  # CJK Extension A
            or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
            or 0xF900 <= code <= 0xFAFF  # CJK Compatibility Ideographs
            or 0xFE30 <= code <= 0xFE4F  # CJK Compatibility Forms
            or 0xFF01 <= code <= 0xFF60  # Fullwidth Latin / punctuation
            or 0xFF61 <= code <= 0xFF9F  # Halfwidth Katakana
            or 0xFFE0 <= code <= 0xFFEF  # Fullwidth signs
        ):
            continue
        out.append(ch)
    result = " ".join("".join(out).split()).strip()
    return result


def _is_renderable_translation(text: str) -> bool:
    """
    Return True if translation should be rendered on-page.

    Only reject empty strings and clear refusals.
    """
    cleaned = " ".join(text.replace("\r", " ").replace("\n", " ").split()).strip()
    if not cleaned:
        return False
    if _looks_like_refusal_text(cleaned):
        return False
    return True


def _looks_like_refusal_text(text: str) -> bool:
    lower = text.lower()
    markers = (
        "i can't provide",
        "i cannot provide",
        "i can't assist",
        "i cannot assist",
        "i'm unable to",
        "i am unable to",
        "sorry, i can't",
        "sorry, i cannot",
    )
    return any(marker in lower for marker in markers)


def _contains_cjk(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if (
            0x3040 <= code <= 0x30FF
            or 0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
        ):
            return True
    return False


def _script_profile(text: str) -> tuple[int, int, int, int]:
    """Return (script_total, kana_count, kanji_count, punct_count)."""
    kana_count = 0
    kanji_count = 0
    punct_count = 0
    script_total = 0

    for ch in text:
        code = ord(ch)
        is_kana = 0x3040 <= code <= 0x30FF
        is_kanji = 0x4E00 <= code <= 0x9FFF
        if is_kana:
            kana_count += 1
            script_total += 1
        elif is_kanji:
            kanji_count += 1
            script_total += 1
        elif ch.isalnum():
            script_total += 1
        elif ch in "…．．。、・,.:;!?！？♡♥❤～〜「」『』（）()[]【】<>＞＜+-*/=＿ー":
            punct_count += 1

    return script_total, kana_count, kanji_count, punct_count


def _looks_like_romaji_noise(text: str) -> bool:
    """Detect transliterated gibberish that should not be rendered as English."""
    words = [w.lower() for w in re.findall(r"[A-Za-z']+", text)]
    if len(words) < 3:
        return False

    common_en = {
        "the",
        "this",
        "that",
        "what",
        "when",
        "where",
        "then",
        "well",
        "okay",
        "right",
        "please",
        "come",
        "again",
        "good",
        "more",
        "with",
        "have",
        "gonna",
        "suck",
        "lick",
        "ouch",
        "slip",
        "gulp",
        "throb",
        "twitch",
        "moan",
        "heh",
        "huh",
        "ah",
        "oh",
        "mmm",
        "yeah",
        "is",
        "it",
        "so",
        "hot",
        "mom",
    }
    english_hits = sum(
        1
        for w in words
        if w in common_en
        or w.endswith(("ing", "ed", "ly", "er", "est", "tion", "ness", "ment", "ful"))
    )
    romaji_hits = sum(
        1
        for w in words
        if len(w) >= 4
        and (
            re.search(r"(sh|ch|ts|ry|ny|ky|gy|ja|ju|jo)", w) is not None
            or w in {"desu", "kun", "chan", "sama", "senpai"}
        )
    )
    vowel_ending = sum(1 for w in words if len(w) >= 3 and w[-1] in "aeiou")

    if romaji_hits >= 2 and english_hits <= max(1, len(words) // 4):
        return True
    if vowel_ending >= max(3, int(len(words) * 0.75)) and english_hits == 0:
        return True
    return False
