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

from .config import Settings, settings
from .detector import TextRegion, detect_text_regions
from .ocr import get_ocr_engine
from .renderer import inpaint_text_region, render_text_on_image
from .translator import TranslationConstraint, translate_texts

logger = logging.getLogger(__name__)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


@dataclass
class RenderTextUnit:
    """A renderable text unit, potentially split from a larger detected region."""

    region: TextRegion
    source_text: str
    style_hint: str
    max_words: int
    max_chars: int
    parent_region_index: int


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
        cv2.imwrite(str(out), image)
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
        max_words = _max_words_for_unit(style_hint, ocr_region.w, ocr_region.h)
        max_chars = _max_chars_for_unit(style_hint, ocr_region.w, ocr_region.h)
        units.append(
            RenderTextUnit(
                region=ocr_region,
                source_text=text,
                style_hint=style_hint,
                max_words=max_words,
                max_chars=max_chars,
                parent_region_index=region_idx,
            )
        )
        inpaint_region_indices.add(region_idx)
        logger.info(
            "  Region %d.%d [%s, max_words=%d, max_chars=%d]: '%s'",
            region_idx + 1,
            sub_idx,
            style_hint,
            max_words,
            max_chars,
            text[:80],
        )

    if not units:
        logger.warning("No text extracted from any region. Saving original image.")
        out = _resolve_output_path(image_path, output_path, cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), image)
        return out

    # 4. Translate
    logger.info("Step 3/4: Translating %d text segments...", len(units))
    source_texts = [unit.source_text for unit in units]
    constraints = [
        TranslationConstraint(
            style=unit.style_hint,
            max_words=unit.max_words,
            max_chars=unit.max_chars,
        )
        for unit in units
    ]
    translated_texts = translate_texts(
        source_texts,
        source_lang=cfg.source_lang,
        model=cfg.translation_model,
        constraints=constraints,
    )
    translated_texts = [
        _sanitize_translated_text(text=text, unit=unit)
        for text, unit in zip(translated_texts, units)
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

    # 5. Inpaint original text, then render translated text
    logger.info("Step 4/4: Inpainting and rendering translated text...")
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
            region = unit.region
            result = render_text_on_image(
                result,
                translated,
                region.x,
                region.y,
                region.w,
                region.h,
                region_mask=region.mask,
                style_hint=unit.style_hint,
            )

    # 6. Save output
    out = _resolve_output_path(image_path, output_path, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), result)
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
    - Local OCR (`USE_MODAL=false`): sequential
    - Modal OCR (`USE_MODAL=true`): bounded parallel workers
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

    if cfg.use_modal and total > 1:
        max_workers = min(cfg.modal_max_parallel_pages, total)
        total_batches = (total + max_workers - 1) // max_workers
        logger.info(
            "Using batched parallel page processing with Modal (%d workers per batch).",
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
    return out_dir / f"{image_path.stem}_translated{image_path.suffix}"


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

    merge_gap = max(8, int(min(region.w, region.h) * 0.14))
    merged_boxes = _merge_nearby_boxes(boxes, gap=merge_gap)
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
    """Coarse style hint for translation constraints."""
    clean = "".join(source_text.split())
    if not clean:
        return "sfx"

    script_total, kana_count, kanji_count, punct_count = _script_profile(clean)
    area = max(1, w * h)
    char_count = len(clean)
    has_sentence_punct = any(c in clean for c in "。！？?!")
    kana_ratio = kana_count / float(max(1, script_total))
    punct_ratio = punct_count / float(max(1, char_count))
    aspect = h / float(max(1, w))

    if char_count <= 2:
        return "sfx"
    if char_count <= 6 and not has_sentence_punct:
        return "sfx"
    if char_count <= 8 and punct_ratio >= 0.28 and kanji_count <= 1:
        return "sfx"
    if char_count <= 10 and aspect >= 1.45 and kana_count >= 3 and kanji_count <= 1:
        return "sfx"
    # Noisy kana-heavy OCR tends to transliterate badly; force concise SFX style.
    if area < 8500 and kana_count >= 8 and kanji_count <= 1 and punct_ratio >= 0.18:
        return "sfx"
    if area < 7000 and kana_ratio >= 0.90 and kanji_count <= 1 and char_count >= 12:
        return "sfx"
    if h < int(w * 0.58) and char_count <= 12:
        return "sfx"
    return "dialogue"


def _max_words_for_unit(style_hint: str, w: int, h: int) -> int:
    """Word-budget heuristic used before rendering."""
    area = max(1, w * h)
    aspect = h / float(max(1, w))
    if style_hint == "sfx":
        if area < 2500:
            return 2
        if area < 7000:
            return 3
        if area < 14000:
            return 4
        return 6

    if area < 2500:
        base = 4
    elif area < 5000:
        base = 6
    elif area < 9000:
        base = 8
    elif area < 14000:
        base = 10
    else:
        base = 13

    if aspect >= 1.45:
        base = min(base, 8)
    if aspect >= 1.90:
        base = min(base, 6)
    return base


def _max_chars_for_unit(style_hint: str, w: int, h: int) -> int:
    """Character-budget heuristic to prevent long transliterated render strings."""
    area = max(1, w * h)
    aspect = h / float(max(1, w))
    if style_hint == "sfx":
        if area < 2500:
            return 12
        if area < 7000:
            return 18
        if area < 14000:
            return 24
        return 34

    if area < 2500:
        base = 26
    elif area < 5000:
        base = 38
    elif area < 9000:
        base = 56
    elif area < 14000:
        base = 72
    else:
        base = 96

    if aspect >= 1.45:
        base = min(base, 56)
    if aspect >= 1.90:
        base = min(base, 40)
    return base


def _apply_word_limit(text: str, max_words: int) -> str:
    """Hard clamp translated text to a max word count before rendering."""
    clean = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if not clean:
        return clean
    if max_words <= 0:
        return clean

    words = clean.split(" ")
    if len(words) <= max_words:
        return clean

    trimmed = " ".join(words[:max_words]).rstrip(".,;:!?")
    return f"{trimmed}..."


def _apply_char_limit(text: str, max_chars: int) -> str:
    """Hard clamp translated text to a max character count before rendering."""
    clean = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if not clean:
        return clean
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean

    clipped = clean[:max_chars].rstrip(".,;:!?- ")
    if not clipped:
        return clean[:max_chars]
    return f"{clipped}..."


def _sanitize_translated_text(text: str, unit: RenderTextUnit) -> str:
    """Clamp and guard against untranslated CJK leakage before rendering."""
    clipped = _apply_char_limit(
        _apply_word_limit(text, unit.max_words),
        unit.max_chars,
    )
    if not _contains_cjk(clipped) and not _looks_like_romaji_noise(clipped):
        return clipped
    # Last-resort fallback: keep output English-only when model returns untranslated text.
    if unit.style_hint == "dialogue":
        return "..."
    return "Ah..."


def _is_renderable_translation(text: str) -> bool:
    """
    Return True if translation should be rendered on-page.

    Reject empty strings and punctuation-only placeholders (e.g. "...").
    """
    cleaned = " ".join(text.replace("\r", " ").replace("\n", " ").split()).strip()
    if not cleaned:
        return False
    if _looks_like_refusal_text(cleaned):
        return False
    return re.search(r"[A-Za-z0-9]", cleaned) is not None


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
