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
import time

import cv2
import numpy as np
from PIL import Image as PILImage

from .config import Settings, settings
from .detector import TextRegion, detect_text_regions
from .ocr import get_ocr_engine
from .renderer import detect_text_color, inpaint_text_region, render_text_on_image
from .translator import TranslationConstraint, translate_texts

import uuid

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
    session_id: str | None = None,
) -> Path:
    """
    Translate a single manga page end-to-end.

    Args:
        image_path: Path to the input manga page image.
        output_path: Path to save the translated image. Auto-generated if None.
        config: Settings override. Uses global settings if None.
        session_id: OpenRouter session ID to pass through to translation API calls.

    Returns:
        Path to the saved translated image.
    """
    cfg = config or settings
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    logger.debug("=" * 60)
    logger.debug("Processing: %s", image_path.name)
    logger.debug("=" * 60)

    # 1. Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    logger.debug("Image loaded: %dx%d", image.shape[1], image.shape[0])

    # 2. Detect text regions using ML model
    logger.debug("Step 1/4: Detecting text regions (ML model)...")
    regions, text_mask = detect_text_regions(image)
    regions = _merge_overlapping_detected_regions(regions, image, text_mask)
    logger.debug("Found %d text regions.", len(regions))

    if not regions:
        logger.warning("No text regions detected. Saving original image.")
        out = _resolve_output_path(image_path, output_path, cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        _save_image(out, image)
        return out

    # 3. OCR - extract text from each region
    logger.debug("Step 2/4: Extracting text (OCR, lang=%s)...", cfg.source_lang)
    ocr_engine = get_ocr_engine(cfg.source_lang)
    units: list[RenderTextUnit] = []
    inpaint_region_indices: set[int] = set()

    # Pre-split all regions into OCR sub-regions
    all_ocr_tasks: list[tuple[int, int, TextRegion]] = []
    for region_idx, region in enumerate(regions):
        ocr_regions = _split_region_for_ocr(region)
        if len(ocr_regions) >= 2:
            logger.debug(
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
        logger.debug(
            "  Running OCR in parallel (%d workers, %d sub-regions)...",
            max_workers,
            len(all_ocr_tasks),
        )
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
        logger.debug(
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

    _assign_render_context_regions(
        units=units,
        parent_regions=regions,
        image=image,
        text_mask=text_mask,
    )

    # 4. Translate
    logger.debug("Step 3/4: Translating %d text segments...", len(units))
    source_texts = [unit.source_text for unit in units]
    constraints = [
        _build_translation_constraint(unit)
        for unit in units
    ]
    translated_texts = translate_texts(
        source_texts,
        source_lang=cfg.source_lang,
        model=cfg.active_translation_model,
        constraints=constraints,
        session_id=session_id,
    )
    # Light sanitization: only block CJK leakage, let renderer handle sizing
    translated_texts = [
        _sanitize_translated_text(text=text)
        for text in translated_texts
    ]
    renderable_unit_mask = [
        _is_renderable_unit(unit, text)
        for unit, text in zip(units, translated_texts)
    ]
    # Never erase text unless we have a replacement that passed the renderability
    # gate. A preserved source bubble is preferable to an empty inpainted bubble.
    inpaintable_unit_mask = list(renderable_unit_mask)
    renderable_region_indices = {
        unit.parent_region_index
        for unit, can_render in zip(units, renderable_unit_mask)
        if can_render
    }
    inpaintable_region_indices = {
        unit.parent_region_index
        for unit, should_inpaint in zip(units, inpaintable_unit_mask)
        if should_inpaint
    }

    for i, (src, tgt) in enumerate(zip(source_texts, translated_texts)):
        logger.debug("  [%d] %s → %s", i + 1, src[:40], tgt[:60])
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
    logger.debug("Step 4/5: Detecting original text colours...")
    for unit, can_render in zip(units, renderable_unit_mask):
        if not can_render:
            continue
        region = unit.region
        unit.text_color = detect_text_color(
            region_image=region.cropped,
            region_mask=region.mask,
        )
        if unit.text_color is not None:
            logger.debug(
                "  Region at (%d,%d): detected colour RGB%s",
                region.x,
                region.y,
                unit.text_color,
            )

    # 6. Inpaint original text, then render translated text
    logger.debug("Step 5/5: Inpainting and rendering translated text...")
    result = image.copy()

    # Count how many renderable units each parent region has.
    # If a parent has multiple renderable children (split-region case),
    # we must NOT expand each sub-region to the full parent bubble —
    # otherwise all sub-translations overlap in the same bubble.
    parent_renderable_counts: dict[int, int] = {}
    parent_unit_counts: dict[int, int] = {}
    for unit, can_render in zip(units, renderable_unit_mask):
        parent_unit_counts[unit.parent_region_index] = (
            parent_unit_counts.get(unit.parent_region_index, 0) + 1
        )
        if can_render:
            parent_renderable_counts[unit.parent_region_index] = (
                parent_renderable_counts.get(unit.parent_region_index, 0) + 1
            )

    # First pass: inpaint only text units with renderable replacements. For split
    # parents, erase only selected child masks so one accepted dialogue line
    # cannot wipe nearby skipped SFX or dialogue.
    for region_idx in sorted(inpaintable_region_indices):
        region = regions[region_idx]
        if parent_unit_counts.get(region_idx, 0) > 1:
            for unit, should_inpaint in zip(units, inpaintable_unit_mask):
                if unit.parent_region_index != region_idx:
                    continue
                if not should_inpaint:
                    continue
                if unit.region.mask is None:
                    continue
                if unit.region.mask.shape[:2] != (unit.region.h, unit.region.w):
                    continue
                result = inpaint_text_region(
                    result,
                    unit.region.x,
                    unit.region.y,
                    unit.region.w,
                    unit.region.h,
                    unit.region.mask,
                )
            continue

        result = inpaint_text_region(
            result, region.x, region.y, region.w, region.h, region.mask
        )

    # Second pass: render translated text onto the cleaned image
    for unit, translated, can_render in zip(units, translated_texts, renderable_unit_mask):
        if can_render:
            # If multiple renderable units share a parent, keep each in its own
            # local render context to avoid overlapping translations in adjacent
            # bubbles/SFX while still giving the renderer enough surrounding
            # pixels to find the bubble outline.
            if parent_renderable_counts.get(unit.parent_region_index, 0) > 1:
                region = unit.context_region or unit.region
                render_mask = _project_region_mask_into_context(
                    unit_region=unit.region,
                    context_region=region,
                )
            else:
                region = unit.context_region or unit.region
                render_mask = _project_region_mask_into_context(
                    unit_region=unit.region,
                    context_region=region,
                )
            rendered_result, did_render = render_text_on_image(
                result,
                translated,
                region.x,
                region.y,
                region.w,
                region.h,
                region_mask=render_mask,
                style_hint=unit.style_hint,
                text_color=unit.text_color,
                allow_bubble_expansion=True,
                return_status=True,
            )
            if did_render:
                result = rendered_result
            else:
                result = _restore_unit_source_text(
                    result=result,
                    original=image,
                    unit=unit,
                )

    # 6. Save output
    out = _resolve_output_path(image_path, output_path, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_image(out, result)
    logger.debug("Saved translated image: %s", out)
    logger.debug("=" * 60)

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
    job_session_id = str(uuid.uuid4())
    logger.debug("Translation job session ID: %s", job_session_id)

    def _translate_one(task: tuple[int, Path]) -> tuple[Path | None, Exception | None]:
        i, img_path = task
        started_at = time.monotonic()
        logger.info("Page %d/%d started: %s", i, total, img_path.name)
        try:
            out = translate_page(img_path, config=cfg, session_id=job_session_id)
            logger.info(
                "Page %d/%d finished: %s -> %s (%.1fs)",
                i,
                total,
                img_path.name,
                out,
                time.monotonic() - started_at,
            )
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


def _restore_unit_source_text(
    result: np.ndarray,
    original: np.ndarray,
    unit: RenderTextUnit,
) -> np.ndarray:
    """Restore original glyph pixels if rendering a translated unit failed."""
    region = unit.region
    img_h, img_w = result.shape[:2]
    x1 = max(0, int(region.x))
    y1 = max(0, int(region.y))
    x2 = min(img_w, int(region.x + region.w))
    y2 = min(img_h, int(region.y + region.h))
    if x2 <= x1 or y2 <= y1:
        return result

    restored = result.copy()
    dst = restored[y1:y2, x1:x2]
    src = original[y1:y2, x1:x2]
    mask = region.mask
    if mask is None or mask.shape[:2] != (region.h, region.w):
        dst[:, :] = src
        return restored

    local_x1 = x1 - region.x
    local_y1 = y1 - region.y
    local_x2 = local_x1 + (x2 - x1)
    local_y2 = local_y1 + (y2 - y1)
    local_mask = mask[local_y1:local_y2, local_x1:local_x2].astype(np.uint8)
    if local_mask.max() > 1:
        _, local_mask = cv2.threshold(local_mask, 127, 255, cv2.THRESH_BINARY)
    else:
        local_mask = (local_mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(local_mask) == 0:
        dst[:, :] = src
        return restored

    local_mask = cv2.dilate(
        local_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    dst[local_mask > 0] = src[local_mask > 0]
    return restored


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


def _assign_render_context_regions(
    units: list[RenderTextUnit],
    parent_regions: list[TextRegion],
    image: np.ndarray,
    text_mask: np.ndarray,
) -> None:
    """Assign bubble-sized render contexts without sharing one huge parent box."""
    if not units:
        return

    units_by_parent: dict[int, list[RenderTextUnit]] = {}
    for unit in units:
        units_by_parent.setdefault(unit.parent_region_index, []).append(unit)

    for parent_idx, sibling_units in units_by_parent.items():
        if parent_idx < 0 or parent_idx >= len(parent_regions):
            continue
        parent_region = parent_regions[parent_idx]

        if len(sibling_units) <= 1:
            unit = sibling_units[0]
            selected = _select_render_context_region(
                unit_region=unit.region,
                parent_region=parent_region,
                style_hint=unit.style_hint,
            )
            if unit.style_hint == "dialogue":
                selected = _expand_single_dialogue_context_region(
                    selected,
                    image=image,
                    text_mask=text_mask,
                )
            unit.context_region = selected
            continue

        sibling_regions = [unit.region for unit in sibling_units]
        for unit in sibling_units:
            unit.context_region = _build_local_render_context_region(
                unit_region=unit.region,
                parent_region=parent_region,
                sibling_regions=sibling_regions,
                image=image,
                text_mask=text_mask,
                style_hint=unit.style_hint,
            )


def _expand_single_dialogue_context_region(
    region: TextRegion,
    image: np.ndarray,
    text_mask: np.ndarray,
) -> TextRegion:
    """Give isolated dialogue enough surrounding pixels to find its balloon."""
    img_h, img_w = image.shape[:2]
    pad_x = min(120, max(22, int(region.w * 0.85), int(region.h * 0.28)))
    pad_y = min(110, max(20, int(region.h * 0.34), int(region.w * 0.22)))
    x1 = max(0, region.x - pad_x)
    y1 = max(0, region.y - pad_y)
    x2 = min(img_w, region.x + region.w + pad_x)
    y2 = min(img_h, region.y + region.h + pad_y)
    if x1 == region.x and y1 == region.y and x2 == region.x + region.w and y2 == region.y + region.h:
        return region

    return TextRegion(
        x=x1,
        y=y1,
        w=x2 - x1,
        h=y2 - y1,
        cropped=image[y1:y2, x1:x2].copy(),
        mask=text_mask[y1:y2, x1:x2].copy(),
    )


def _build_local_render_context_region(
    unit_region: TextRegion,
    parent_region: TextRegion,
    sibling_regions: list[TextRegion],
    image: np.ndarray,
    text_mask: np.ndarray,
    style_hint: str,
) -> TextRegion:
    """Expand a split OCR unit into a local, sibling-aware render context."""
    img_h, img_w = image.shape[:2]
    parent_x1 = max(0, parent_region.x)
    parent_y1 = max(0, parent_region.y)
    parent_x2 = min(img_w, parent_region.x + parent_region.w)
    parent_y2 = min(img_h, parent_region.y + parent_region.h)

    ux1 = unit_region.x
    uy1 = unit_region.y
    ux2 = unit_region.x + unit_region.w
    uy2 = unit_region.y + unit_region.h
    ucx = (ux1 + ux2) * 0.5
    ucy = (uy1 + uy2) * 0.5

    expand_x = max(14, int(unit_region.w * 0.85), int(parent_region.w * 0.04))
    expand_y = max(14, int(unit_region.h * 0.55), int(parent_region.h * 0.04))
    if unit_region.h > unit_region.w * 1.45:
        expand_x = max(expand_x, min(90, int(unit_region.h * 0.20)))
    if unit_region.w > unit_region.h * 1.8:
        expand_y = max(expand_y, min(70, int(unit_region.w * 0.16)))

    x1 = max(parent_x1, ux1 - expand_x)
    y1 = max(parent_y1, uy1 - expand_y)
    x2 = min(parent_x2, ux2 + expand_x)
    y2 = min(parent_y2, uy2 + expand_y)

    left_limit = parent_x1
    right_limit = parent_x2
    top_limit = parent_y1
    bottom_limit = parent_y2
    for sibling in sibling_regions:
        if sibling is unit_region:
            continue

        sx1 = sibling.x
        sy1 = sibling.y
        sx2 = sibling.x + sibling.w
        sy2 = sibling.y + sibling.h
        scx = (sx1 + sx2) * 0.5
        scy = (sy1 + sy2) * 0.5

        y_overlap = max(0, min(uy2, sy2) - max(uy1, sy1))
        x_overlap = max(0, min(ux2, sx2) - max(ux1, sx1))
        if y_overlap >= max(8, int(min(unit_region.h, sibling.h) * 0.18)):
            boundary = int(round((ucx + scx) * 0.5))
            if scx < ucx:
                left_limit = max(left_limit, boundary)
            elif scx > ucx:
                right_limit = min(right_limit, boundary)

        if x_overlap >= max(8, int(min(unit_region.w, sibling.w) * 0.18)):
            boundary = int(round((ucy + scy) * 0.5))
            if scy < ucy:
                top_limit = max(top_limit, boundary)
            elif scy > ucy:
                bottom_limit = min(bottom_limit, boundary)

    x1 = max(x1, left_limit)
    x2 = min(x2, right_limit)
    y1 = max(y1, top_limit)
    y2 = min(y2, bottom_limit)

    min_w = min(parent_region.w, max(unit_region.w + 8, 28))
    min_h = min(parent_region.h, max(unit_region.h + 8, 24))
    if (x2 - x1) < min_w:
        half = min_w // 2
        x1 = max(parent_x1, int(round(ucx)) - half)
        x2 = min(parent_x2, x1 + min_w)
        x1 = max(parent_x1, x2 - min_w)
    if (y2 - y1) < min_h:
        half = min_h // 2
        y1 = max(parent_y1, int(round(ucy)) - half)
        y2 = min(parent_y2, y1 + min_h)
        y1 = max(parent_y1, y2 - min_h)

    if style_hint == "dialogue" and unit_region.h > unit_region.w * 1.35:
        current_w = x2 - x1
        target_w = min(
            parent_region.w,
            max(
                current_w,
                96,
                unit_region.w + min(140, int(unit_region.h * 0.42)),
            ),
        )
        if target_w > current_w:
            half = int(target_w // 2)
            candidate_x1 = max(parent_x1, int(round(ucx)) - half)
            candidate_x2 = min(parent_x2, candidate_x1 + int(target_w))
            candidate_x1 = max(parent_x1, candidate_x2 - int(target_w))
            candidate = image[int(y1) : int(y2), int(candidate_x1) : int(candidate_x2)]
            if _crop_has_neutral_dialogue_surface(candidate):
                x1, x2 = candidate_x1, candidate_x2

    if x2 <= x1 or y2 <= y1:
        return unit_region

    return TextRegion(
        x=int(x1),
        y=int(y1),
        w=int(x2 - x1),
        h=int(y2 - y1),
        cropped=image[int(y1) : int(y2), int(x1) : int(x2)].copy(),
        mask=text_mask[int(y1) : int(y2), int(x1) : int(x2)].copy(),
    )


def _crop_has_neutral_dialogue_surface(crop: np.ndarray) -> bool:
    """Return True for white/grey speech-balloon-like local crops."""
    if crop is None or crop.size == 0:
        return False
    if crop.shape[0] < 24 or crop.shape[1] < 24:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    neutral_ratio = float(np.mean(hsv[:, :, 1] < 85))
    median_sat = float(np.median(hsv[:, :, 1]))
    mean_gray = float(np.mean(gray))
    return neutral_ratio >= 0.72 and median_sat <= 70 and mean_gray >= 115


def _build_translation_constraint(unit: RenderTextUnit) -> TranslationConstraint:
    """Derive concise translation budgets from the region most likely to be rendered."""
    render_region = unit.context_region or unit.region
    rw = render_region.w
    rh = render_region.h

    # Context expansion gives the renderer surrounding pixels for balloon
    # detection, but it should not make the translation budget panel-wide when
    # the original Japanese glyphs were a narrow vertical column.
    if unit.style_hint == "dialogue" and render_region is not unit.region:
        source_aspect = unit.region.h / float(max(1, unit.region.w))
        if source_aspect >= 1.20 and unit.region.w < 220:
            rw = min(
                rw,
                max(
                    56,
                    int(unit.region.w * 1.45),
                    unit.region.w + min(64, int(unit.region.h * 0.16)),
                ),
            )
            rh = min(rh, max(48, int(unit.region.h * 1.40)))

    area = max(1, int(rw * rh))
    aspect = rh / float(max(1, rw))

    if unit.style_hint == "sfx":
        max_chars = 12 if area < 12000 else 16
        if rw >= 160 or rh >= 220:
            max_chars = min(18, max_chars + 2)
        return TranslationConstraint(
            style="sfx",
            max_words=3,
            max_chars=max_chars,
        )

    # Very narrow regions (w < 60px) can only fit ~3-4 chars per line at min
    # readable font size (12px). Cap aggressively to avoid tiny unreadable text.
    if rw < 60:
        return TranslationConstraint(style="dialogue", max_words=5, max_chars=22)
    if rw < 80:
        return TranslationConstraint(style="dialogue", max_words=7, max_chars=28)
    if rw < 100:
        return TranslationConstraint(style="dialogue", max_words=8, max_chars=34)

    # For normal-sized bubbles, use generous budgets — the renderer will handle
    # fitting. Only constrain tightly for very tall/narrow or very small regions.
    max_words = 14
    max_chars = 60
    if aspect >= 1.75:
        max_words = 8 if rw < 120 or area < 24000 else 10
        max_chars = 32 if rw < 120 else 44
    elif aspect >= 1.35:
        max_words = 9 if area < 26000 else 12
        max_chars = 38 if area < 26000 else 50
    elif area < 18000:
        max_words = 10
        max_chars = 44
    elif area < 32000:
        max_words = 12
        max_chars = 52

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

    region_area = region.w * region.h
    large_mixed_region = region_area >= 120_000 and len(boxes) >= 10

    bright_bubble_region = (
        not large_mixed_region and _context_region_has_bright_bubble(region)
    )
    if bright_bubble_region:
        bubble_text_boxes, remaining_boxes = _split_bright_bubble_text_groups(
            region=region,
            boxes=boxes,
        )
        if len(bubble_text_boxes) >= 2:
            merged_boxes = bubble_text_boxes + _merge_nearby_boxes(
                remaining_boxes,
                gap=max(8, int(min(region.w, region.h) * 0.04)),
            )
            merged_boxes = _refine_oversized_ocr_groups(
                merged_boxes=merged_boxes,
                component_boxes=boxes,
                region_w=region.w,
                region_h=region.h,
            )
        else:
            clustered_boxes = _split_bright_bubble_by_text_clusters(
                region=region,
                boxes=boxes,
            )
            if len(clustered_boxes) >= 2:
                merged_boxes = clustered_boxes
            else:
                # A detector region lying on one balloon is already one semantic
                # dialogue unit. Splitting its vertical glyph columns produces
                # one-character translations and leaves most of the balloon blank.
                return [region]
    else:
        merged_boxes = []

    if large_mixed_region:
        # Panel-sized detections often contain several speech bubbles plus SFX.
        # The regular phrase-merging pass can chain those scattered components
        # into one giant OCR unit, which later renders as one enormous paragraph.
        merge_gap = max(8, int(min(region.w, region.h) * 0.025))
    else:
        merge_gap = max(8, int(min(region.w, region.h) * 0.06))

    if not large_mixed_region and region.h > region.w * 2.5:
        # Tall regions often contain multiple vertical text columns in one bubble.
        # Use a more generous merge gap so adjacent columns stay as one OCR unit.
        merge_gap = max(merge_gap, int(max(region.w, region.h) * 0.05))

    if large_mixed_region:
        bubble_text_boxes, remaining_boxes = _split_bright_bubble_text_groups(
            region=region,
            boxes=boxes,
        )
        sfx_boxes = _merge_nearby_boxes(remaining_boxes, gap=merge_gap)
        merged_boxes = bubble_text_boxes + sfx_boxes
        merged_boxes = _refine_oversized_ocr_groups(
            merged_boxes=merged_boxes,
            component_boxes=boxes,
            region_w=region.w,
            region_h=region.h,
        )
    elif not bright_bubble_region:
        merged_boxes = _merge_nearby_boxes(boxes, gap=merge_gap)

    if not large_mixed_region and not bright_bubble_region:
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
    max_split_count = 18 if large_mixed_region else 8
    if len(merged_boxes) > max_split_count:
        merged_boxes = _merge_nearby_boxes(
            merged_boxes,
            gap=max(12, int(min(region.w, region.h) * (0.045 if large_mixed_region else 0.20))),
        )
    if len(merged_boxes) > max_split_count and large_mixed_region:
        for factor in (0.07, 0.10, 0.14):
            candidate = _merge_nearby_boxes(
                merged_boxes,
                gap=max(14, int(min(region.w, region.h) * factor)),
            )
            if 2 <= len(candidate) <= max_split_count:
                merged_boxes = candidate
                break
    if len(merged_boxes) < 2 or len(merged_boxes) > max_split_count:
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


def _split_bright_bubble_text_groups(
    region: TextRegion,
    boxes: list[tuple[int, int, int, int]],
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Separate text inside bright speech bubbles from nearby outlined SFX."""
    if not boxes or region.cropped is None or region.cropped.size == 0:
        return [], boxes

    gray = cv2.cvtColor(region.cropped, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return [], boxes

    bright_threshold = int(np.clip(np.percentile(gray, 82), 205, 235))
    _, bright = cv2.threshold(gray, bright_threshold, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=2,
    )
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bright,
        connectivity=8,
    )
    region_area = max(1, region.w * region.h)
    candidates: list[tuple[int, int, int, int, int, int]] = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        bbox_area = max(1, w * h)
        fill_ratio = area / float(bbox_area)
        edge_touches = int(x <= 1) + int(y <= 1)
        edge_touches += int((x + w) >= region.w - 1)
        edge_touches += int((y + h) >= region.h - 1)

        if area < max(900, int(region_area * 0.004)):
            continue
        if w < 42 or h < 42:
            continue
        if bbox_area > int(region_area * 0.70):
            continue
        if edge_touches >= 2 and bbox_area > int(region_area * 0.18):
            continue
        if fill_ratio < 0.34:
            continue
        candidates.append((x, y, x + w, y + h, area, label))

    if not candidates:
        return [], boxes

    candidates.sort(key=lambda item: item[4], reverse=True)
    used: set[int] = set()
    bubble_groups: list[tuple[int, int, int, int]] = []
    for cx1, cy1, cx2, cy2, area, label in candidates:
        members: list[tuple[int, int, int, int]] = []
        for idx, box in enumerate(boxes):
            if idx in used:
                continue
            bx1, by1, bx2, by2 = box
            bcx = (bx1 + bx2) * 0.5
            bcy = (by1 + by2) * 0.5
            ix1, iy1 = max(cx1, bx1), max(cy1, by1)
            ix2, iy2 = min(cx2, bx2), min(cy2, by2)
            overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            box_area = max(1, (bx2 - bx1) * (by2 - by1))
            center_inside = cx1 - 3 <= bcx <= cx2 + 3 and cy1 - 3 <= bcy <= cy2 + 3
            if center_inside or overlap >= int(box_area * 0.30):
                members.append(box)

        if not members:
            continue
        if len(members) < 2 and area < 6000:
            continue

        lobe_groups = _split_bright_component_into_text_lobes(
            component_mask=(labels == label).astype(np.uint8) * 255,
            component_box=(cx1, cy1, cx2, cy2),
            members=members,
        )
        if len(lobe_groups) >= 2:
            for group_members in lobe_groups:
                gx1 = min(item[0] for item in group_members)
                gy1 = min(item[1] for item in group_members)
                gx2 = max(item[2] for item in group_members)
                gy2 = max(item[3] for item in group_members)
                if (gx2 - gx1) < 12 or (gy2 - gy1) < 12:
                    continue
                bubble_groups.append((gx1, gy1, gx2, gy2))
            for idx, box in enumerate(boxes):
                if box in members:
                    used.add(idx)
            continue

        gx1 = min(item[0] for item in members)
        gy1 = min(item[1] for item in members)
        gx2 = max(item[2] for item in members)
        gy2 = max(item[3] for item in members)
        if (gx2 - gx1) < 12 or (gy2 - gy1) < 12:
            continue

        bubble_groups.append((gx1, gy1, gx2, gy2))
        for idx, box in enumerate(boxes):
            if box in members:
                used.add(idx)

    remaining = [box for idx, box in enumerate(boxes) if idx not in used]
    return bubble_groups, remaining


def _split_bright_component_into_text_lobes(
    component_mask: np.ndarray,
    component_box: tuple[int, int, int, int],
    members: list[tuple[int, int, int, int]],
) -> list[list[tuple[int, int, int, int]]]:
    """Split touching speech-balloon paper into separate text lobe groups."""
    if component_mask.size == 0 or len(members) < 2:
        return []

    cx1, cy1, cx2, cy2 = component_box
    cw = max(1, cx2 - cx1)
    ch = max(1, cy2 - cy1)
    if cw < 90 and ch < 90:
        return []

    crop = component_mask[cy1:cy2, cx1:cx2].astype(np.uint8)
    comp_area = cv2.countNonZero(crop)
    if comp_area < 1200:
        return []

    # Erosion exposes the neck between adjacent balloons while usually keeping
    # a single balloon as one component. The selected lobes are dilated back
    # before text boxes are assigned to avoid losing edge glyphs.
    k = _odd(max(7, min(35, int(round(min(cw, ch) * 0.13)))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(crop, kernel, iterations=1)
    if cv2.countNonZero(eroded) < max(220, int(comp_area * 0.16)):
        return []

    num_lobes, lobe_labels, stats, _ = cv2.connectedComponentsWithStats(
        eroded,
        connectivity=8,
    )
    lobes: list[tuple[int, int, int, int, int, np.ndarray]] = []
    for label in range(1, num_lobes):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(180, int(comp_area * 0.055)):
            continue
        if w < 22 or h < 22:
            continue
        lobe = (lobe_labels == label).astype(np.uint8) * 255
        lobe = cv2.dilate(lobe, kernel, iterations=1)
        lobe = cv2.bitwise_and(lobe, crop)
        if cv2.countNonZero(lobe) < max(220, int(comp_area * 0.07)):
            continue
        bbox = _mask_bbox_local(lobe)
        if bbox is None:
            continue
        lx1, ly1, lx2, ly2 = bbox
        lobes.append((lx1, ly1, lx2, ly2, cv2.countNonZero(lobe), lobe))

    if len(lobes) < 2:
        return []

    groups: list[list[tuple[int, int, int, int]]] = [[] for _ in lobes]
    for box in members:
        bx1, by1, bx2, by2 = box
        local_box = (
            max(0, bx1 - cx1),
            max(0, by1 - cy1),
            min(cw, bx2 - cx1),
            min(ch, by2 - cy1),
        )
        if local_box[2] <= local_box[0] or local_box[3] <= local_box[1]:
            continue

        bcx = (local_box[0] + local_box[2]) * 0.5
        bcy = (local_box[1] + local_box[3]) * 0.5
        best_idx = -1
        best_score = -1e9
        box_area = max(1, (local_box[2] - local_box[0]) * (local_box[3] - local_box[1]))
        for idx, (lx1, ly1, lx2, ly2, lobe_area, lobe_mask) in enumerate(lobes):
            crop_mask = lobe_mask[
                local_box[1] : local_box[3],
                local_box[0] : local_box[2],
            ]
            overlap = cv2.countNonZero(crop_mask)
            inside = lx1 - 3 <= bcx <= lx2 + 3 and ly1 - 3 <= bcy <= ly2 + 3
            lcx = (lx1 + lx2) * 0.5
            lcy = (ly1 + ly2) * 0.5
            dist = abs(lcx - bcx) + abs(lcy - bcy)
            score = overlap * 12.0 + (500.0 if inside else 0.0) - dist
            if overlap >= int(box_area * 0.08) and score > best_score:
                best_idx = idx
                best_score = score

        if best_idx >= 0:
            groups[best_idx].append(box)

    groups = [group for group in groups if group]
    if len(groups) < 2:
        return []

    centers: list[tuple[float, float]] = []
    for group in groups:
        gx1 = min(item[0] for item in group)
        gy1 = min(item[1] for item in group)
        gx2 = max(item[2] for item in group)
        gy2 = max(item[3] for item in group)
        centers.append(((gx1 + gx2) * 0.5, (gy1 + gy2) * 0.5))

    separated_pairs = 0
    for i, (ax, ay) in enumerate(centers):
        for bx, by in centers[i + 1 :]:
            if abs(ax - bx) >= max(28, cw * 0.16) or abs(ay - by) >= max(28, ch * 0.16):
                separated_pairs += 1
    if separated_pairs == 0:
        return []

    return groups


def _split_bright_bubble_by_text_clusters(
    region: TextRegion,
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Split side-by-side balloons when text clusters have a clear empty gap."""
    if len(boxes) < 4 or region.w < 110 or region.h < 70:
        return []

    gap = max(14, int(min(region.w, region.h) * 0.085))
    clusters = _merge_nearby_boxes(boxes, gap=gap)
    clusters = [
        box
        for box in clusters
        if (box[2] - box[0]) >= 18
        and (box[3] - box[1]) >= 18
        and ((box[2] - box[0]) * (box[3] - box[1])) >= 300
    ]
    if not (2 <= len(clusters) <= 5):
        return []

    clusters = sorted(clusters, key=lambda b: (b[0] + b[2]) * 0.5)
    clear_gap = False
    for left, right in zip(clusters, clusters[1:]):
        x_gap = right[0] - left[2]
        y_overlap = max(0, min(left[3], right[3]) - max(left[1], right[1]))
        min_h = max(1, min(left[3] - left[1], right[3] - right[1]))
        if x_gap >= max(38, int(region.w * 0.20)) and y_overlap >= int(min_h * 0.20):
            clear_gap = True
            break

    if not clear_gap:
        clusters_y = sorted(clusters, key=lambda b: (b[1] + b[3]) * 0.5)
        for upper, lower in zip(clusters_y, clusters_y[1:]):
            y_gap = lower[1] - upper[3]
            x_overlap = max(0, min(upper[2], lower[2]) - max(upper[0], lower[0]))
            min_w = max(1, min(upper[2] - upper[0], lower[2] - lower[0]))
            if y_gap >= max(38, int(region.h * 0.20)) and x_overlap >= int(min_w * 0.20):
                clear_gap = True
                break

    if not clear_gap:
        return []

    return clusters


def _mask_bbox_local(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _odd(value: int) -> int:
    """Round up to an odd integer for morphology kernels."""
    return value if value % 2 == 1 else value + 1


def _refine_oversized_ocr_groups(
    merged_boxes: list[tuple[int, int, int, int]],
    component_boxes: list[tuple[int, int, int, int]],
    region_w: int,
    region_h: int,
) -> list[tuple[int, int, int, int]]:
    """Split oversized OCR groups that still span several speech bubbles."""
    if not merged_boxes or not component_boxes:
        return merged_boxes

    refined: list[tuple[int, int, int, int]] = []
    region_area = max(1, region_w * region_h)
    for group in merged_boxes:
        gx1, gy1, gx2, gy2 = group
        gw = gx2 - gx1
        gh = gy2 - gy1
        group_area = max(1, gw * gh)

        if (
            group_area < max(95_000, int(region_area * 0.22))
            or gw < 260
            or gh < 180
        ):
            refined.append(group)
            continue

        members: list[tuple[int, int, int, int]] = []
        for box in component_boxes:
            bx1, by1, bx2, by2 = box
            bcx = (bx1 + bx2) * 0.5
            bcy = (by1 + by2) * 0.5
            ix1, iy1 = max(gx1, bx1), max(gy1, by1)
            ix2, iy2 = min(gx2, bx2), min(gy2, by2)
            overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            box_area = max(1, (bx2 - bx1) * (by2 - by1))
            center_inside = gx1 <= bcx <= gx2 and gy1 <= bcy <= gy2
            if center_inside or overlap >= int(box_area * 0.50):
                members.append(box)

        if len(members) < 4:
            refined.append(group)
            continue

        tight_gap = max(8, int(min(region_w, region_h) * 0.018))
        split = _merge_nearby_boxes(members, gap=tight_gap)
        split = [
            b
            for b in split
            if (b[2] - b[0]) >= 12
            and (b[3] - b[1]) >= 12
            and ((b[2] - b[0]) * (b[3] - b[1])) >= 140
        ]

        if 2 <= len(split) <= 10:
            refined.extend(split)
        else:
            refined.append(group)

    return refined


def _merge_aligned_ocr_boxes(
    boxes: list[tuple[int, int, int, int]],
    region_w: int,
    region_h: int,
) -> list[tuple[int, int, int, int]]:
    """Merge OCR boxes that are clearly one phrase split across tight columns/rows."""
    if len(boxes) < 2:
        return boxes

    x_gap_limit = max(12, int(region_w * 0.10))
    y_gap_limit = max(10, int(region_h * 0.05))

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
            and x_gap <= min(x_gap_limit, int(max(aw, bw) * 0.30) + 6)
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
    has_dialogue_comma = any(c in clean for c in "、,")

    if has_sentence_punct or kanji_count > 0 or kana_count >= 2:
        if has_dialogue_comma and kana_count >= 3:
            return "dialogue"
        if has_sentence_punct and kana_count >= 4:
            return "dialogue"
        if _is_effect_like_source(clean, kana_count=kana_count, kanji_count=kanji_count):
            return "sfx"
        return "dialogue"
    if script_total == 0 or char_count <= 2:
        return "sfx"
    return "dialogue"


def _is_effect_like_source(clean: str, kana_count: int, kanji_count: int) -> bool:
    """Detect compact kana effects by shape, not by manga-specific word lists."""
    if kanji_count > 0 or kana_count == 0:
        return False

    kana = "".join(ch for ch in clean if _is_kana_char(ch))
    if not kana or len(kana) != kana_count:
        return False

    if kana_count == 1:
        return True

    if kana_count <= 8 and any(ch in clean for ch in "っッぁぃぅぇぉァィゥェォー〜～"):
        return True

    return kana_count <= 8 and re.search(r"([\u3040-\u30ff])\1", kana) is not None



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


def _is_renderable_unit(unit: RenderTextUnit, text: str) -> bool:
    """
    Return True if a translated unit should be rendered on-page.

    Combines basic text checks with context-aware filtering for SFX
    fragments — tiny regions with single-character source/translated text
    that are typically noise from individual Japanese characters detected
    as separate regions.
    """
    if not _is_renderable_translation(text):
        return False

    cleaned = text.strip()
    region = unit.region
    area = max(1, region.w * region.h)

    if unit.style_hint == "sfx":
        src_clean = unit.source_text.strip()
        src_compact = "".join(src_clean.split())
        script_total, kana_count, kanji_count, punct_count = _script_profile(src_compact)
        meaningful_chars = len(src_compact)
        context = unit.context_region or unit.region
        bubble_interjection = (
            kana_count >= 1
            and punct_count >= 2
            and re.search(r"[A-Za-z]", cleaned) is not None
            and (
                _context_region_has_bright_bubble(unit.region)
                or _context_region_has_bright_bubble(context)
                or _crop_has_neutral_dialogue_surface(unit.region.cropped)
                or _crop_has_neutral_dialogue_surface(context.cropped)
            )
        )
        if script_total == 0:
            return False
        if (
            meaningful_chars >= 3
            and (script_total / float(meaningful_chars)) < 0.50
            and not bubble_interjection
        ):
            return False
        if (kana_count + kanji_count) < 2 and re.search(r"[A-Za-z0-9]", src_compact):
            return False

        # Skip very short SFX translations in small regions (fragment noise).
        if len(cleaned) <= 2 and area < 4000:
            return False
        # Skip single-character source text in small SFX regions — these are
        # individual Japanese characters (e.g. 'お', 'ん') that shouldn't be
        # rendered as separate tiny SFX.
        if len(src_clean) <= 1 and area < 8000:
            return False
        # Skip numeric/symbol-only SFX (e.g. "13", "1/", "1:")
        if re.fullmatch(r'[\d\s/.:;,!?\-()]+', cleaned):
            return False
        if not _is_compact_sfx_render(cleaned):
            return False
        if (
            not bubble_interjection
            and not _is_effect_like_source(src_compact, kana_count=kana_count, kanji_count=kanji_count)
        ):
            return False

    if unit.style_hint == "dialogue" and _is_short_nonbubble_dialogue_noise(unit, cleaned):
        return False

    if _is_large_nonbubble_translation_mismatch(unit, cleaned):
        return False

    return True


def _is_large_nonbubble_translation_mismatch(unit: RenderTextUnit, translated: str) -> bool:
    """Avoid destroying large decorative/source blocks for tiny bad replacements.

    A common OCR failure on cover/title/SFX overlays is that a large Japanese
    block is recognized as a small interjection. Rendering that tiny fragment
    after inpainting the whole source block looks worse than preserving the
    original. This gate is deliberately limited to non-bubble regions.
    """
    clean_translation = " ".join(translated.split())
    if not clean_translation:
        return True

    region = unit.context_region or unit.region
    region_area = max(1, region.w * region.h)
    if region_area < 42_000:
        return False
    if _context_region_has_bright_bubble(unit.region) or _context_region_has_bright_bubble(region):
        return False

    source_compact = "".join(unit.source_text.split())
    if len(source_compact) < 8:
        return False

    script_total, kana_count, kanji_count, _ = _script_profile(source_compact)
    if script_total < 8:
        return False

    translated_words = re.findall(r"[A-Za-z0-9']+", clean_translation)
    translated_alpha_len = sum(len(word) for word in translated_words)
    source_is_semantic_block = kanji_count >= 2 or len(source_compact) >= 14
    if source_is_semantic_block and region_area >= 70_000:
        return True

    replacement_is_tiny = (
        len(translated_words) <= 2
        and translated_alpha_len <= 14
        and not any(ch in clean_translation for ch in ".?!")
    )
    if source_is_semantic_block and replacement_is_tiny:
        return True

    # Large kana-only effects can still be valid SFX, but if OCR translated a
    # long effect block to a one-word whisper, preserve the original rather than
    # leaving a half-erased page.
    if kanji_count == 0 and kana_count >= 10 and translated_alpha_len <= 8:
        return True

    return False


def _is_compact_sfx_render(text: str) -> bool:
    """Return True when translated SFX is short enough to render as lettering."""
    clean = text.strip()
    if not clean:
        return False

    tokens = re.findall(r"[A-Za-z]+", clean)
    if not tokens:
        return False

    letters = "".join(tokens)
    if len(tokens) > 2 or not (2 <= len(letters) <= 14):
        return False

    has_separator = any(not ch.isalnum() and not ch.isspace() for ch in clean)
    has_repeated_letter = re.search(r"([A-Za-z])\1", letters) is not None
    if len(tokens) > 1:
        return has_separator
    return has_repeated_letter or len(letters) <= 3


def _is_short_nonbubble_dialogue_noise(unit: RenderTextUnit, translated: str) -> bool:
    """Suppress short dialogue-looking OCR fragments from large SFX/art regions."""

    src_clean = "".join(unit.source_text.split())
    if not src_clean:
        return True

    script_total, kana_count, kanji_count, _ = _script_profile(src_clean)
    alpha_words = re.findall(r"[A-Za-z']+", translated)
    word_count = len(alpha_words)
    translated_len = len(translated.strip())
    region_area = max(1, unit.region.w * unit.region.h)
    context = unit.context_region or unit.region

    short_source = (
        script_total <= 4
        or (kana_count <= 4 and kanji_count == 0)
        or (script_total <= 8 and kanji_count == 0)
    )
    short_translation = translated_len <= 16 or word_count <= 3
    large_source_region = region_area >= 12_000 or (
        context.w * context.h >= 35_000 and region_area >= 5_000
    )
    if not alpha_words:
        return True

    # Preserve short, ordinary dialogue when the OCR glyph crop itself sits on
    # speech-balloon paper.  A large expanded context can contain some white
    # bubble elsewhere, which previously let free-text fragments render on art.
    if _context_region_has_bright_bubble(unit.region):
        return False

    source_has_sentence_punct = any(c in src_clean for c in "。！？?!")
    if (
        kanji_count == 0
        and not source_has_sentence_punct
        and word_count <= 8
        and translated_len <= 60
    ):
        return True

    if (
        kanji_count == 0
        and word_count <= 3
        and translated_len <= 18
        and region_area >= 30_000
    ):
        return True

    if not (short_source and short_translation and large_source_region):
        return False

    return translated_len <= 10 or word_count <= 2


def _context_region_has_bright_bubble(region: TextRegion) -> bool:
    """Return True when the local context contains a filled bright bubble area."""
    if region.cropped is None or region.cropped.size == 0:
        return False
    if region.w < 24 or region.h < 24:
        return False

    gray = cv2.cvtColor(region.cropped, cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return False
    threshold = int(np.clip(np.percentile(gray, 82), 205, 235))
    _, bright = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
        bright,
        connectivity=8,
    )
    hsv = cv2.cvtColor(region.cropped, cv2.COLOR_BGR2HSV)

    context_area = max(1, region.w * region.h)
    for label in range(1, num_labels):
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        bbox_area = max(1, w * h)
        fill_ratio = area / float(bbox_area)
        if area < max(500, int(context_area * 0.12)):
            continue
        if bbox_area < int(context_area * 0.10):
            continue
        component_mask = _labels == label
        mean_sat = float(np.mean(hsv[:, :, 1][component_mask]))
        if mean_sat > 55:
            continue
        if fill_ratio >= 0.50:
            return True
    return False


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


def _is_kana_char(ch: str) -> bool:
    code = ord(ch)
    return 0x3040 <= code <= 0x30FF


def _merge_overlapping_detected_regions(
    regions: list[TextRegion],
    image: np.ndarray,
    text_mask: np.ndarray,
) -> list[TextRegion]:
    """Merge detected regions that overlap significantly (likely the same bubble)."""
    if len(regions) < 2:
        return regions

    def _iou(a: TextRegion, b: TextRegion) -> float:
        ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
        bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / float(max(1, union))

    def _contains_centroid(container: TextRegion, other: TextRegion) -> bool:
        cx = other.x + other.w // 2
        cy = other.y + other.h // 2
        return (
            container.x <= cx < container.x + container.w
            and container.y <= cy < container.y + container.h
        )

    def _is_vertical_column(r: TextRegion) -> bool:
        return r.h > r.w * 2.0 or (r.h > r.w * 1.5 and r.w < 200)

    def _are_adjacent_columns(a: TextRegion, b: TextRegion) -> bool:
        if not (_is_vertical_column(a) or _is_vertical_column(b)):
            return False
        h_gap = max(0, max(a.x, b.x) - min(a.x + a.w, b.x + b.w))
        max_h_gap = max(12, int(min(a.w, b.w) * 0.50))
        if h_gap > max_h_gap:
            return False
        y_overlap = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
        min_h = min(a.h, b.h)
        if y_overlap < int(min_h * 0.25):
            return False
        return True

    merged = regions.copy()
    changed = True
    while changed:
        changed = False
        next_regions: list[TextRegion] = []
        used = [False] * len(merged)
        for i, a in enumerate(merged):
            if used[i]:
                continue
            used[i] = True
            x1, y1, x2, y2 = a.x, a.y, a.x + a.w, a.y + a.h

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                b = merged[j]
                iou = _iou(a, b)
                should_merge = (
                    iou > 0.20
                    or (_contains_centroid(a, b) and iou > 0)
                    or (_contains_centroid(b, a) and iou > 0)
                    or _are_adjacent_columns(a, b)
                )
                if not should_merge:
                    continue
                x1, y1 = min(x1, b.x), min(y1, b.y)
                x2, y2 = max(x2, b.x + b.w), max(y2, b.y + b.h)
                used[j] = True
                changed = True

            nx, ny = max(0, x1), max(0, y1)
            nx2 = min(image.shape[1], x2)
            ny2 = min(image.shape[0], y2)
            if nx2 > nx and ny2 > ny:
                next_regions.append(
                    TextRegion(
                        x=nx,
                        y=ny,
                        w=nx2 - nx,
                        h=ny2 - ny,
                        cropped=image[ny:ny2, nx:nx2].copy(),
                        mask=text_mask[ny:ny2, nx:nx2].copy(),
                    )
                )
        merged = next_regions

    return merged
