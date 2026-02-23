"""
Modal services for manga-ocr inference (GPU and CPU variants).

Deploy with:
    modal deploy modal_ocr_service.py

This deploys both classes under the same app:
- GPU class: `modal.Cls.from_name("manga-ocr-service", "MangaOCR")`
- CPU class: `modal.Cls.from_name("manga-ocr-service", "MangaOCRCPU")`
"""

from __future__ import annotations

import modal

app = modal.App("manga-ocr-service")

# Build a container image with all dependencies manga-ocr needs.
manga_ocr_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "manga-ocr>=0.1.12",
        "Pillow>=10.0",
        "opencv-python-headless>=4.9",
    )
)


_MODAL_SECRETS = [modal.Secret.from_name("huggingface-secret")]


class _MangaOCRServiceBase:
    """Shared inference logic for GPU/CPU Modal OCR services."""

    _FORCE_CPU = False

    @modal.enter()
    def load_model(self) -> None:
        from manga_ocr import MangaOcr

        self.model = MangaOcr(force_cpu=self._FORCE_CPU)

    @modal.method()
    def ocr(self, image_bytes: bytes) -> str:
        """
        Run manga-ocr on a PNG/JPEG image.

        Args:
            image_bytes: Raw image bytes (PNG or JPEG encoded).

        Returns:
            Extracted text string.
        """
        import io

        from PIL import Image

        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        try:
            text = self.model(pil_image)
            return text.strip()
        except Exception as e:
            print(f"manga-ocr inference failed: {e}")
            return ""


@app.cls(
    gpu="T4",
    image=manga_ocr_image,
    secrets=_MODAL_SECRETS,
    scaledown_window=120,  # Keep warm for 2 min after last call
    timeout=300,  # 5 min max per call
)
class MangaOCR(_MangaOCRServiceBase):
    """
    Runs manga-ocr inference on a T4 GPU.

    The model is loaded once when the container starts via @modal.enter(),
    and reused across all subsequent calls.
    """


@app.cls(
    cpu=2,
    memory=8192,
    image=manga_ocr_image,
    secrets=_MODAL_SECRETS,
    scaledown_window=120,  # Keep warm for 2 min after last call
    timeout=300,  # 5 min max per call
)
class MangaOCRCPU(_MangaOCRServiceBase):
    """
    Runs manga-ocr inference on Modal CPU workers.

    This is slower than GPU, but significantly cheaper to keep online.
    """

    _FORCE_CPU = True
