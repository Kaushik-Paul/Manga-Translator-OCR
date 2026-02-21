"""
OCR engines for extracting text from manga bubbles.

- MangaOCREngine: Uses `manga-ocr` library, optimized for Japanese manga text.
- EasyOCREngine: Uses `easyocr`, supports Chinese and Japanese.

Both engines run in CPU-only mode.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

logger = logging.getLogger(__name__)


class OCREngine(ABC):
    """Base class for OCR engines."""

    @abstractmethod
    def extract_text(self, image: NDArray, region_mask: NDArray | None = None) -> str:
        """
        Extract text from a cropped bubble image.

        Args:
            image: Cropped bubble image in BGR format (numpy array).
            region_mask: Optional per-pixel text mask aligned with `image`.

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
    _COMMON_JP_PATTERNS = (
        "です",
        "ます",
        "ない",
        "する",
        "して",
        "それ",
        "これ",
        "お",
        "あ",
        "ん",
    )

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

    def _run_manga_ocr(self, image: NDArray) -> str:
        """Run manga-ocr on a BGR image crop."""
        self._ensure_model()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        try:
            text = self._model(pil_image)  # type: ignore[misc]
            return text.strip()
        except Exception as e:
            logger.warning("manga-ocr failed on region: %s", e)
            return ""

    def _crop_with_mask(self, image: NDArray, region_mask: NDArray | None) -> NDArray:
        """Tighten the OCR crop using the detector mask when available."""
        if region_mask is None or region_mask.shape[:2] != image.shape[:2]:
            return image

        if region_mask.max() > 1:
            _, binary = cv2.threshold(region_mask, 127, 255, cv2.THRESH_BINARY)
        else:
            binary = (region_mask > 0).astype(np.uint8) * 255

        non_zero = cv2.countNonZero(binary)
        if non_zero < 20:
            return image

        x, y, w, h = cv2.boundingRect(binary)
        pad_x = max(8, int(w * 0.35))
        pad_y = max(8, int(h * 0.45))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image.shape[1], x + w + pad_x)
        y2 = min(image.shape[0], y + h + pad_y)
        return image[y1:y2, x1:x2].copy()

    def _is_noisy_text(self, text: str) -> bool:
        """Heuristic for OCR text that likely needs multi-pass refinement."""
        t = text.strip()
        if not t:
            return True

        invalid = sum(1 for c in t if not _is_common_ocr_char(c))
        punctuation = sum(1 for c in t if c in "…．．。、・,.:;!?！？♡♥❤～〜")
        kana = sum(1 for c in t if _is_kana(c))
        kanji = sum(1 for c in t if _is_kanji(c))

        if invalid > max(2, len(t) // 8):
            return True
        if len(t) >= 24 and kanji == 0 and kana > 12 and punctuation > 6:
            return True
        if len(t) >= 18 and punctuation > len(t) * 0.35:
            return True
        return False

    def _score_candidate(self, text: str) -> float:
        """Score OCR candidate quality using lightweight Japanese-text heuristics."""
        t = text.strip()
        if not t:
            return -1e9

        score = 0.0
        kanji = sum(1 for c in t if _is_kanji(c))
        kana = sum(1 for c in t if _is_kana(c))
        punctuation = sum(1 for c in t if c in "…．．。、・,.:;!?！？♡♥❤～〜")
        invalid = sum(1 for c in t if not _is_common_ocr_char(c))

        score += kanji * 2.0 + kana * 1.2
        score -= punctuation * 0.35
        score -= invalid * 3.0

        for token in self._COMMON_JP_PATTERNS:
            score += t.count(token) * 1.7

        repeated = len(re.findall(r"(.)\1{4,}", t))
        score -= repeated * 2.5

        if "。" in t or "、" in t:
            score += 1.0
        if len(t) < 2:
            score -= 2.0
        return score

    def _build_preprocess_variants(self, image: NDArray) -> list[NDArray]:
        """Generate a small set of OCR-friendly image variants."""
        variants: list[NDArray] = []

        variants.append(cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        variants.append(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR))

        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR))

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharp = cv2.filter2D(image, -1, kernel)
        variants.append(sharp)

        return variants

    def extract_text(self, image: NDArray, region_mask: NDArray | None = None) -> str:
        crop = self._crop_with_mask(image, region_mask)
        base = self._run_manga_ocr(crop)
        if not base:
            return ""

        area = crop.shape[0] * crop.shape[1]
        should_refine = area >= 50000 or self._is_noisy_text(base)
        if not should_refine:
            return base.strip()

        candidates = [base]
        for variant in self._build_preprocess_variants(crop):
            text = self._run_manga_ocr(variant)
            if text:
                candidates.append(text)

        best = max(candidates, key=self._score_candidate)
        return best.strip()


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

    def extract_text(self, image: NDArray, region_mask: NDArray | None = None) -> str:
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


def _is_kanji(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _is_kana(char: str) -> bool:
    return ("\u3040" <= char <= "\u30ff") or char == "ー"


def _is_common_ocr_char(char: str) -> bool:
    if char.isspace():
        return True
    if _is_kanji(char) or _is_kana(char):
        return True
    if char.isdigit():
        return True
    return char in "…．．。、・,.:;!?！？♡♥❤～〜「」『』（）()[]【】<>＞＜+-*/=＿ー"
