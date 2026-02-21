"""Manga Translator OCR - Doujinshi/Manga page translator."""

from .pipeline import translate_directory, translate_page

__all__ = ["translate_page", "translate_directory"]
