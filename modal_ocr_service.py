"""
Modal service for manga-ocr GPU inference.

Deploy with:
    modal deploy modal_ocr_service.py

This creates a remotely-hosted GPU service that runs manga-ocr on a T4 GPU.
The local client calls it via `modal.Cls.from_name("manga-ocr-service", "MangaOCR")`.
"""

import modal

app = modal.App("manga-ocr-service")

# Build a container image with all the dependencies manga-ocr needs.
manga_ocr_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "manga-ocr>=0.1.12",
        "Pillow>=10.0",
        "opencv-python-headless>=4.9",
    )
)


@app.cls(
    gpu="T4",
    image=manga_ocr_image,
    scaledown_window=120,  # Keep warm for 2 min after last call
    timeout=300,  # 5 min max per call
)
class MangaOCR:
    """
    Runs manga-ocr inference on a T4 GPU.

    The model is loaded once when the container starts via @modal.enter(),
    and reused across all subsequent calls.
    """

    @modal.enter()
    def load_model(self):
        from manga_ocr import MangaOcr

        self.model = MangaOcr()

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
