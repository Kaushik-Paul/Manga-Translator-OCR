"""
Pipeline orchestrator - ties the full translation workflow together.

Flow: Load image → Detect text (ML model) → OCR → Translate → Inpaint + Render → Save
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
from numpy.typing import NDArray

from .config import Settings, settings
from .detector import TextRegion, detect_text_regions
from .ocr import get_ocr_engine
from .renderer import inpaint_text_region, render_text_on_image
from .translator import translate_texts

logger = logging.getLogger(__name__)


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
    extracted_texts: list[str] = []
    for i, region in enumerate(regions):
        text = ocr_engine.extract_text(region.cropped, region.mask)
        extracted_texts.append(text)
        if text:
            logger.info("  Region %d: '%s'", i + 1, text[:80])
        else:
            logger.info("  Region %d: (no text detected)", i + 1)

    # Filter to only regions that have text
    text_regions: list[tuple[TextRegion, str]] = [
        (r, t) for r, t in zip(regions, extracted_texts) if t.strip()
    ]

    if not text_regions:
        logger.warning("No text extracted from any region. Saving original image.")
        out = _resolve_output_path(image_path, output_path, cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), image)
        return out

    # 4. Translate
    logger.info("Step 3/4: Translating %d text segments...", len(text_regions))
    source_texts = [t for _, t in text_regions]
    translated_texts = translate_texts(
        source_texts,
        source_lang=cfg.source_lang,
        model=cfg.translation_model,
    )

    for i, (src, tgt) in enumerate(zip(source_texts, translated_texts)):
        logger.info("  [%d] %s → %s", i + 1, src[:40], tgt[:60])

    # 5. Inpaint original text, then render translated text
    logger.info("Step 4/4: Inpainting and rendering translated text...")
    result = image.copy()

    # First pass: inpaint all text regions (clean the original text)
    for region, _ in text_regions:
        result = inpaint_text_region(
            result, region.x, region.y, region.w, region.h, region.mask
        )

    # Second pass: render translated text onto the cleaned image
    for (region, _), translated in zip(text_regions, translated_texts):
        if translated.strip():
            result = render_text_on_image(
                result,
                translated,
                region.x,
                region.y,
                region.w,
                region.h,
                region_mask=region.mask,
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

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    # Find image files
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    image_files = sorted(
        f
        for f in input_dir.iterdir()
        if f.suffix.lower() in image_extensions and f.is_file()
    )

    if not image_files:
        logger.warning("No image files found in %s", input_dir)
        return []

    logger.info("Found %d images to translate in %s", len(image_files), input_dir)

    results: list[Path] = []
    for i, img_path in enumerate(image_files, 1):
        logger.info("\n[%d/%d] Processing %s...", i, len(image_files), img_path.name)
        try:
            out = translate_page(img_path, config=cfg)
            results.append(out)
        except Exception as e:
            logger.error("Failed to process %s: %s", img_path.name, e)

    logger.info("\nDone! Translated %d/%d images.", len(results), len(image_files))
    return results


def _resolve_output_path(
    image_path: Path, output_path: str | Path | None, cfg: Settings
) -> Path:
    """Determine the output file path."""
    if output_path:
        return Path(output_path)
    out_dir = cfg.output_dir
    return out_dir / f"{image_path.stem}_translated{image_path.suffix}"
