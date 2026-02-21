"""
Text detection using the comic-text-detector ONNX model.

Uses the ML model from https://github.com/dmMaze/comic-text-detector
(same model used by manga-image-translator) for accurate text detection
in manga/comic pages. Runs on CPU via ONNX Runtime.

The model outputs:
- Text block bounding boxes (groups of text)
- Text line segments
- Text mask (pixel-level text segmentation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Model configuration
_MODEL_REPO = "mayocream/comic-text-detector-onnx"
_MODEL_FILE = "comic-text-detector.onnx"
_MODEL_INPUT_SIZE = (1024, 1024)  # Model expects 1024x1024 input


@dataclass
class TextRegion:
    """A detected text region with its bounding box and cropped image."""

    x: int
    y: int
    w: int
    h: int
    cropped: NDArray  # The cropped region image (BGR)
    mask: NDArray | None = None  # Per-pixel text mask for this region

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


class ComicTextDetector:
    """
    ML-based text detector using the comic-text-detector ONNX model.

    Singleton pattern — model loaded once and reused.
    """

    _instance: ComicTextDetector | None = None
    _session = None

    def __new__(cls) -> ComicTextDetector:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_model(self) -> None:
        """Download and load the ONNX model if not already loaded."""
        if self._session is not None:
            return

        logger.info("Loading comic-text-detector ONNX model...")

        from huggingface_hub import hf_hub_download
        import onnxruntime as ort

        model_path = hf_hub_download(
            repo_id=_MODEL_REPO,
            filename=_MODEL_FILE,
        )
        logger.info("Model downloaded to: %s", model_path)

        # Create ONNX Runtime session (CPU only)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        logger.info("comic-text-detector model loaded (CPU).")

    def detect(
        self,
        image: NDArray,
        text_threshold: float = 0.5,
        min_area: int = 100,
        padding: int = 5,
    ) -> tuple[list[TextRegion], NDArray]:
        """
        Detect text regions in a manga page.

        Args:
            image: Input image (BGR, as loaded by cv2.imread).
            text_threshold: Threshold for text mask binarization.
            min_area: Minimum region area in pixels.
            padding: Padding around detected regions.

        Returns:
            Tuple of (list of TextRegion, full-page text mask).
        """
        self._ensure_model()

        h, w = image.shape[:2]

        # Preprocess: resize to model input size
        input_tensor = self._preprocess(image)

        # Run inference
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: input_tensor})

        # The model outputs: blks (bounding boxes), mask, lines
        # Output structure depends on model version, but typically:
        # outputs[0] = text block bounding boxes or mask
        # outputs[1] = text segmentation mask (if available)

        # Parse the text mask from model outputs
        text_mask = self._parse_mask(outputs, h, w, text_threshold)

        # Extract text regions from the mask
        regions = self._extract_regions_from_mask(
            image, text_mask, min_area=min_area, padding=padding
        )

        # Sort in manga reading order
        regions = _sort_manga_order(regions, page_width=w)

        logger.info(
            "Detected %d text regions using comic-text-detector.", len(regions)
        )
        return regions, text_mask

    def _preprocess(self, image: NDArray) -> NDArray:
        """Preprocess image for the ONNX model."""
        # Resize to model input size
        resized = cv2.resize(image, _MODEL_INPUT_SIZE)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] and transpose to NCHW format
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)  # Add batch dim -> NCHW

        return blob

    def _parse_mask(
        self,
        outputs: list,
        orig_h: int,
        orig_w: int,
        threshold: float,
    ) -> NDArray:
        """Parse the text segmentation mask from model outputs."""
        # The model can output different things depending on version.
        # We look for the mask output (2D or 3D array representing
        # per-pixel text probability)
        mask = None

        for i, out in enumerate(outputs):
            arr = np.array(out)
            logger.debug("Output %d shape: %s, dtype: %s", i, arr.shape, arr.dtype)

            # The mask output is typically a 2D or has spatial dimensions
            # matching the input size (1024x1024)
            if len(arr.shape) >= 2:
                # Check if any dimension matches model input size
                flat = arr.squeeze()
                if len(flat.shape) >= 2:
                    spatial_dims = flat.shape[-2:]
                    if (
                        spatial_dims[0] >= 256
                        and spatial_dims[1] >= 256
                    ):
                        # This looks like a spatial mask
                        if len(flat.shape) == 3:
                            # Take the text channel (usually index 0 or last)
                            flat = flat[0] if flat.shape[0] <= 4 else flat[-1]
                        mask = flat
                        logger.debug("Using output %d as text mask (shape: %s)", i, mask.shape)
                        break

        if mask is None:
            # Fallback: use the largest spatial output
            best_size = 0
            for out in outputs:
                arr = np.array(out).squeeze()
                if len(arr.shape) >= 2:
                    size = arr.shape[-1] * arr.shape[-2]
                    if size > best_size:
                        best_size = size
                        mask = arr if len(arr.shape) == 2 else arr[0]

        if mask is None:
            logger.warning("Could not find text mask in model outputs, falling back to empty mask.")
            return np.zeros((orig_h, orig_w), dtype=np.uint8)

        # Normalize mask to [0, 1]
        if mask.max() > 1.0:
            mask = mask / mask.max()

        # Resize mask back to original image dimensions
        mask_resized = cv2.resize(
            mask.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )

        # Binarize
        binary_mask = (mask_resized > threshold).astype(np.uint8) * 255

        return binary_mask

    def _extract_regions_from_mask(
        self,
        image: NDArray,
        mask: NDArray,
        min_area: int = 400,
        padding: int = 8,
    ) -> list[TextRegion]:
        """Extract text regions from the binary mask, merging nearby text into blocks."""
        h, w = image.shape[:2]

        # Aggressively merge nearby text into speech-bubble-sized blocks
        # Step 1: Close small gaps between characters
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)

        # Step 2: Dilate aggressively to merge text within the same bubble
        # Use a large kernel to connect text lines that belong together
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        cleaned = cv2.dilate(cleaned, kernel_dilate, iterations=3)

        # Step 3: Close again to fill any remaining gaps
        kernel_close2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close2, iterations=2)

        # Find contours on the merged mask
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions: list[TextRegion] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            rx, ry, rw, rh = cv2.boundingRect(contour)

            # Skip very tiny regions (likely noise)
            if rw < 15 or rh < 15:
                continue

            # Apply padding
            x1 = max(0, rx - padding)
            y1 = max(0, ry - padding)
            x2 = min(w, rx + rw + padding)
            y2 = min(h, ry + rh + padding)

            # Extract the ORIGINAL mask for this region (for inpainting)
            region_mask = mask[y1:y2, x1:x2].copy()
            cropped = image[y1:y2, x1:x2].copy()

            regions.append(
                TextRegion(
                    x=x1,
                    y=y1,
                    w=x2 - x1,
                    h=y2 - y1,
                    cropped=cropped,
                    mask=region_mask,
                )
            )

        return regions


def detect_text_regions(
    image: NDArray,
    **kwargs,
) -> tuple[list[TextRegion], NDArray]:
    """
    Detect text regions in a manga page using the ML model.

    Args:
        image: Input image (BGR format).

    Returns:
        Tuple of (list of TextRegion, full-page text mask).
    """
    detector = ComicTextDetector()
    return detector.detect(image, **kwargs)


def _sort_manga_order(
    regions: list[TextRegion], page_width: int, num_columns: int = 3
) -> list[TextRegion]:
    """
    Sort regions in manga reading order (right-to-left, top-to-bottom).
    """
    if not regions:
        return regions

    col_width = page_width / num_columns

    def sort_key(r: TextRegion) -> tuple[int, int]:
        col = int(r.center[0] / col_width)
        col = num_columns - 1 - col  # Reverse for right-to-left
        return (col, r.center[1])

    return sorted(regions, key=sort_key)
