"""
OCR engines for extracting text from manga bubbles.

- MangaOCREngine: Uses `manga-ocr` library, optimized for Japanese manga text.
- EasyOCREngine: Uses `easyocr`, supports Chinese and Japanese.

Both engines run in CPU-only mode.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

logger = logging.getLogger(__name__)


class OCREngine(ABC):
    """Base class for OCR engines."""

    @abstractmethod
    def extract_text(self, image: NDArray) -> str:
        """
        Extract text from a cropped bubble image.

        Args:
            image: Cropped bubble image in BGR format (numpy array).

        Returns:
            Extracted text string, or empty string if nothing found.
        """
        ...


class MangaOCREngine(OCREngine):
    """
    Japanese manga OCR using the `manga-ocr` library.

    Handles vertical/horizontal text, furigana, varied fonts.
    Model is lazy-loaded on first use (~400MB download on first run).
    """

    _instance: MangaOCREngine | None = None
    _model = None

    def __new__(cls) -> MangaOCREngine:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_model(self) -> None:
        if self._model is None:
            logger.info("Loading manga-ocr model (first time may download ~400MB)...")
            from manga_ocr import MangaOcr

            self._model = MangaOcr()
            logger.info("manga-ocr model loaded.")

    def extract_text(self, image: NDArray) -> str:
        self._ensure_model()
        # manga-ocr expects a PIL Image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        try:
            text = self._model(pil_image)  # type: ignore[misc]
            return text.strip()
        except Exception as e:
            logger.warning("manga-ocr failed on region: %s", e)
            return ""


class EasyOCREngine(OCREngine):
    """
    Multi-language OCR using EasyOCR.

    Supports Chinese (Simplified + Traditional) and Japanese.
    Runs in CPU mode (gpu=False).
    """

    _instances: dict[str, EasyOCREngine] = {}
    _reader = None
    _lang: str = "zh"

    def __new__(cls, lang: str = "zh") -> EasyOCREngine:
        if lang not in cls._instances:
            instance = super().__new__(cls)
            instance._lang = lang
            cls._instances[lang] = instance
        return cls._instances[lang]

    def __init__(self, lang: str = "zh") -> None:
        self._lang = lang

    def _ensure_reader(self) -> None:
        if self._reader is None:
            logger.info(
                "Loading EasyOCR reader for '%s' (first time may download models)...",
                self._lang,
            )
            import easyocr

            # Map our language codes to EasyOCR language codes
            lang_map = {
                "zh": ["ch_sim", "ch_tra", "en"],
                "ja": ["ja", "en"],
            }
            langs = lang_map.get(self._lang, [self._lang, "en"])
            self._reader = easyocr.Reader(langs, gpu=False)
            logger.info("EasyOCR reader loaded.")

    def extract_text(self, image: NDArray) -> str:
        self._ensure_reader()
        try:
            results = self._reader.readtext(image, detail=0)  # type: ignore[union-attr]
            # Join all detected text fragments
            text = " ".join(results).strip()
            return text
        except Exception as e:
            logger.warning("EasyOCR failed on region: %s", e)
            return ""


def get_ocr_engine(lang: str = "ja") -> OCREngine:
    """
    Factory function to get the appropriate OCR engine.

    Args:
        lang: Source language - "ja" for Japanese (uses manga-ocr),
              "zh" for Chinese (uses EasyOCR).

    Returns:
        An OCR engine instance.
    """
    if lang == "ja":
        return MangaOCREngine()
    elif lang == "zh":
        return EasyOCREngine(lang="zh")
    else:
        logger.warning("Unknown language '%s', falling back to EasyOCR.", lang)
        return EasyOCREngine(lang=lang)
