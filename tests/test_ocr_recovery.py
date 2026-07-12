from __future__ import annotations

import numpy as np

from main.ocr import MangaOCREngine


class _EmptyBaseRecoveringOCR(MangaOCREngine):
    """Deterministic fake that only recognizes an enhanced OCR variant."""

    def __new__(cls):
        return object.__new__(cls)

    def __init__(self) -> None:
        self.calls = 0

    def _run_manga_ocr(self, image: np.ndarray) -> str:
        self.calls += 1
        return "日本語" if self.calls == 3 else ""


def test_empty_base_ocr_retries_preprocessed_variants() -> None:
    engine = _EmptyBaseRecoveringOCR()
    image = np.full((80, 60, 3), 255, dtype=np.uint8)

    text = engine.extract_text(image)

    assert text == "日本語"
    assert engine.calls > 1


class _AlwaysEmptyOCR(_EmptyBaseRecoveringOCR):
    def _run_manga_ocr(self, image: np.ndarray) -> str:
        self.calls += 1
        return ""


def test_empty_base_and_variants_still_return_empty() -> None:
    engine = _AlwaysEmptyOCR()
    image = np.full((80, 60, 3), 255, dtype=np.uint8)

    assert engine.extract_text(image) == ""
    assert engine.calls == 5
