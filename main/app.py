"""
Manga Translator OCR - Gradio Web UI Launcher.

Usage:
    uv run python main/app.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path for direct execution
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def setup_logging() -> None:
    """Configure logging for the Gradio app."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy third-party libraries
    for lib in (
        "httpx", "httpcore", "h2", "hpack",
        "PIL", "transformers", "torch",
        "gradio", "modal", "grpc", "grpclib",
        "urllib3", "onnxruntime",
    ):
        logging.getLogger(lib).setLevel(logging.WARNING)


def launch() -> None:
    """Create and launch the Gradio application."""
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("🎌 Launching Manga Translator OCR - Gradio UI")

    from main.ui.gradio_app import create_app

    app, theme, css = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=theme,
        css=css,
        ssr_mode=False,
    )


if __name__ == "__main__":
    launch()
