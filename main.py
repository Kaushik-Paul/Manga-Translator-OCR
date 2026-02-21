"""
Manga Translator OCR - Entry Point

Usage:
    uv run python main.py --input <image_or_directory> [--output <dir>] [--lang ja|zh]
"""

from main.app import run

if __name__ == "__main__":
    run()
