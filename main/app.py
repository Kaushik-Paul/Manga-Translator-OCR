"""
CLI entry point for the Manga Translator.

Usage:
    uv run python main.py --input <image_or_directory> [--output <dir>] [--lang ja|zh] [--model <model>]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import settings
from .pipeline import translate_directory, translate_page


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("easyocr").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="manga-translator",
        description="Translate manga/doujinshi pages from Japanese/Chinese to English",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate a single page (Japanese)
  uv run python main.py --input page01.jpg

  # Translate a directory of Chinese manga pages
  uv run python main.py --input ./manga_pages/ --lang zh --output ./translated/

  # Use a specific model
  uv run python main.py --input page.png --model mistralai/mistral-large-latest
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input image file or directory containing manga pages",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./output",
        help="Output directory for translated images (default: ./output)",
    )
    parser.add_argument(
        "--lang",
        "-l",
        choices=["ja", "zh"],
        default="ja",
        help="Source language: 'ja' for Japanese (default), 'zh' for Chinese",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="OpenRouter model override (default: deepseek/deepseek-chat)",
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
    settings.source_lang = args.lang
    settings.output_dir = Path(args.output)
    if args.model:
        settings.translation_model = args.model

    # Validate
    try:
        settings.validate()
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)

    logger.info("Source language: %s", settings.source_lang)
    logger.info("Translation model: %s", settings.translation_model)
    logger.info("Output directory: %s", settings.output_dir)
    logger.info("─" * 40)

    input_path = Path(args.input)

    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    if input_path.is_dir():
        results = translate_directory(input_path, config=settings)
        logger.info(
            "\n✅ Completed! Translated %d images to %s",
            len(results),
            settings.output_dir,
        )
    elif input_path.is_file():
        result = translate_page(input_path, config=settings)
        logger.info("\n✅ Completed! Translated image saved to: %s", result)
    else:
        logger.error("Input is neither a file nor directory: %s", input_path)
        sys.exit(1)
