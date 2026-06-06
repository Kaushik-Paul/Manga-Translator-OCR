"""
CLI entry point for the Manga Translator.

Usage:
    uv run python -m main.cli --input <image_or_directory> [<image_or_directory> ...]
    [--output <dir>] [--model <model>]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import settings
from .pipeline import is_image_file, list_image_files, translate_images


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy third-party libraries
    for lib in (
        "httpx", "httpcore", "h2", "hpack",
        "PIL", "transformers", "torch",
        "modal", "grpc", "grpclib",
        "urllib3", "onnxruntime",
    ):
        logging.getLogger(lib).setLevel(logging.WARNING)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="manga-translator",
        description="Translate manga/doujinshi pages from Japanese to English",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate a single page
  uv run python -m main.cli --input page01.jpg

  # Translate multiple pages
  uv run python -m main.cli --input page01.jpg page02.jpg page03.webp

  # Translate a directory of pages
  uv run python -m main.cli --input ./manga_pages/ --output ./translated/

  # Use a specific model for the active provider
  uv run python -m main.cli --input page.png --model deepseek-v4-flash
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        nargs="+",
        metavar="PATH",
        help="One or more image files and/or directories",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Output directory for translated images (default: ./output)",
    )

    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Model override for the active translation provider",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose/debug logging",
    )

    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    args = parse_args(argv)
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("🎌 Manga Translator OCR")
    logger.info("─" * 40)

    # Apply CLI args to settings
    settings.output_dir = Path(args.output)
    if args.model:
        settings.set_active_translation_model(args.model)

    # Validate
    try:
        settings.validate()
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)

    logger.info("Source language: %s", settings.source_lang)
    logger.info("Translation provider: %s", settings.translation_provider)
    logger.info("Translation model: %s", settings.active_translation_model)
    logger.info(
        "Detection backend: %s",
        "modal service" if settings.use_detection_model else "local model",
    )
    if settings.use_modal:
        ocr_backend = (
            "modal cpu service"
            if settings.use_mangaocr_cpu
            else "modal gpu service"
        )
    else:
        ocr_backend = "local manga-ocr (cpu)"
    logger.info("OCR backend: %s", ocr_backend)
    logger.info("Output directory: %s", settings.output_dir)
    logger.info("─" * 40)

    candidate_images: list[Path] = []
    for raw_path in args.input:
        input_path = Path(raw_path)
        if not input_path.exists():
            logger.error("Input path does not exist: %s", input_path)
            sys.exit(1)

        if input_path.is_dir():
            dir_images = list_image_files(input_path)
            if not dir_images:
                logger.warning("No image files found in %s", input_path)
            candidate_images.extend(dir_images)
            continue

        if input_path.is_file():
            if not is_image_file(input_path):
                logger.error("Unsupported image extension: %s", input_path)
                sys.exit(1)
            candidate_images.append(input_path)
            continue

        logger.error("Input is neither a file nor directory: %s", input_path)
        sys.exit(1)

    # Remove duplicate paths while preserving order.
    unique_images: list[Path] = []
    seen: set[str] = set()
    for image_path in candidate_images:
        key = str(image_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_images.append(image_path)

    if not unique_images:
        logger.error("No valid input images were found.")
        sys.exit(1)

    if settings.use_modal and len(unique_images) > 1:
        logger.info(
            "USE_MODAL=true -> processing pages in parallel (up to %d workers).",
            settings.modal_max_parallel_pages,
        )
    elif len(unique_images) > 1:
        logger.info("USE_MODAL=false -> processing pages sequentially.")

    results = translate_images(unique_images, config=settings)
    if len(unique_images) == 1 and results:
        logger.info("\n✅ Completed! Translated image saved to: %s", results[0])
    else:
        logger.info(
            "\n✅ Completed! Translated %d/%d images to %s",
            len(results),
            len(unique_images),
            settings.output_dir,
        )

if __name__ == "__main__":
    run()
