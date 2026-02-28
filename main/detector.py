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
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Model configuration
_MODEL_REPO = "mayocream/comic-text-detector-onnx"
_MODEL_FILE = "comic-text-detector.onnx"
_MODEL_PATH_ENV = "COMIC_TEXT_DETECTOR_MODEL_PATH"
_LOCAL_MODEL_PATH = (
    Path(__file__).resolve().parent / "weights" / "comic-text-detector" / _MODEL_FILE
)
_MODEL_INPUT_SIZE = (1024, 1024)  # Model expects 1024x1024 input
_MODAL_DETECTOR_APP = "manga-detector-service"
_MODAL_DETECTOR_CLASS = "ComicTextDetectorService"


def _odd(value: int) -> int:
    """Round up to the nearest odd integer."""
    return value if value % 2 == 1 else value + 1


def _boxes_are_near(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int
) -> bool:
    """Return True if two boxes overlap or are within `gap` pixels."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
    y_gap = max(0, max(ay1, by1) - min(ay2, by2))
    return x_gap <= gap and y_gap <= gap


def _merge_nearby_boxes(
    boxes: list[tuple[int, int, int, int]], gap: int
) -> list[tuple[int, int, int, int]]:
    """Merge overlapping/nearby boxes into larger blocks."""
    if not boxes:
        return []

    merged = boxes.copy()
    changed = True
    while changed:
        changed = False
        next_boxes: list[tuple[int, int, int, int]] = []
        used = [False] * len(merged)

        for i, base in enumerate(merged):
            if used[i]:
                continue
            bx1, by1, bx2, by2 = base
            used[i] = True

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if _boxes_are_near((bx1, by1, bx2, by2), merged[j], gap):
                    ox1, oy1, ox2, oy2 = merged[j]
                    bx1 = min(bx1, ox1)
                    by1 = min(by1, oy1)
                    bx2 = max(bx2, ox2)
                    by2 = max(by2, oy2)
                    used[j] = True
                    changed = True

            next_boxes.append((bx1, by1, bx2, by2))

        merged = next_boxes

    return merged


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
    _session_lock = Lock()

    def __new__(cls) -> ComicTextDetector:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_model(self) -> None:
        """Load bundled ONNX model (with fallback download) if not already loaded."""
        if self._session is not None:
            return
        with self._session_lock:
            if self._session is not None:
                return

            logger.info("Loading comic-text-detector ONNX model...")

            import onnxruntime as ort

            env_model_path = os.getenv(_MODEL_PATH_ENV, "").strip()
            if env_model_path:
                env_candidate = Path(env_model_path).expanduser()
                if not env_candidate.is_file():
                    raise FileNotFoundError(
                        f"{_MODEL_PATH_ENV} points to a missing file: {env_candidate}"
                    )
                model_path = str(env_candidate)
                logger.info(
                    "Using detector model from %s: %s",
                    _MODEL_PATH_ENV,
                    model_path,
                )
            elif _LOCAL_MODEL_PATH.is_file():
                model_path = str(_LOCAL_MODEL_PATH)
                logger.info("Using bundled detector model: %s", model_path)
            else:
                from huggingface_hub import hf_hub_download

                model_path = hf_hub_download(
                    repo_id=_MODEL_REPO,
                    filename=_MODEL_FILE,
                )
                logger.info("Detector model downloaded to: %s", model_path)

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
        text_threshold: float = 0.25,
        min_area: int = 30,
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

        # Merge nearby characters into dialogue-sized blocks with dynamic kernels.
        # Fixed, very-large kernels tend to over-merge across nearby bubbles/panels.
        short_side = min(h, w)
        close_size = _odd(max(5, int(round(short_side * 0.006))))
        dilate_size = _odd(max(11, int(round(short_side * 0.018))))
        close2_size = _odd(max(9, int(round(short_side * 0.014))))

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)
        )
        cleaned = cv2.dilate(cleaned, kernel_dilate, iterations=1)

        kernel_close2 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close2_size, close2_size)
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close2, iterations=1)

        # Find contours on the merged mask
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidate_boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            rx, ry, rw, rh = cv2.boundingRect(contour)
            if rw < 8 or rh < 8:
                continue
            candidate_boxes.append((rx, ry, rx + rw, ry + rh))

        merge_gap = max(10, int(short_side * 0.016))
        merged_boxes = _merge_nearby_boxes(candidate_boxes, gap=merge_gap)

        regions: list[TextRegion] = []
        page_area = h * w
        max_region_area = int(page_area * 0.10)
        max_region_w = int(w * 0.45)
        max_region_h = int(h * 0.45)

        for x1, y1, x2, y2 in merged_boxes:
            rx, ry = x1, y1
            rw = x2 - x1
            rh = y2 - y1

            if (rw * rh) > 25000:
                disconnected_boxes = self._split_disconnected_region(
                    mask=mask,
                    region_bbox=(rx, ry, rw, rh),
                    min_area=min_area,
                    padding=max(2, padding // 2),
                    page_w=w,
                    page_h=h,
                )
                if len(disconnected_boxes) >= 2:
                    for sx1, sy1, sx2, sy2 in disconnected_boxes:
                        region = self._make_region(image, mask, sx1, sy1, sx2, sy2)
                        if self._is_region_viable(region):
                            regions.append(region)
                    continue

            is_overmerged = (
                (rw * rh) > max_region_area
                or rw > max_region_w
                or rh > max_region_h
            )

            if is_overmerged:
                split_boxes = self._split_overmerged_region(
                    mask=mask,
                    region_bbox=(rx, ry, rw, rh),
                    min_area=min_area,
                    padding=max(2, padding // 2),
                    page_w=w,
                    page_h=h,
                )
                if len(split_boxes) >= 2:
                    logger.debug(
                        "Split oversized region (%d,%d,%d,%d) into %d sub-regions.",
                        rx,
                        ry,
                        rw,
                        rh,
                        len(split_boxes),
                    )
                    for x1, y1, x2, y2 in split_boxes:
                        region = self._make_region(image, mask, x1, y1, x2, y2)
                        if self._is_region_viable(region):
                            regions.append(region)
                    continue

            x1 = max(0, rx - padding)
            y1 = max(0, ry - padding)
            x2 = min(w, rx + rw + padding)
            y2 = min(h, ry + rh + padding)
            region = self._make_region(image, mask, x1, y1, x2, y2)
            if self._is_region_viable(region):
                regions.append(region)

        return regions

    def _split_overmerged_region(
        self,
        mask: NDArray,
        region_bbox: tuple[int, int, int, int],
        min_area: int,
        padding: int,
        page_w: int,
        page_h: int,
    ) -> list[tuple[int, int, int, int]]:
        """
        Split a very large merged region using the original (non-merged) text mask.
        """
        rx, ry, rw, rh = region_bbox
        sub_mask = mask[ry : ry + rh, rx : rx + rw]
        if cv2.countNonZero(sub_mask) == 0:
            return []

        short_side = max(1, min(rw, rh))
        close_size = _odd(max(3, int(round(short_side * 0.02))))
        dilate_size = _odd(max(5, int(round(short_side * 0.035))))

        refined = cv2.morphologyEx(
            sub_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size)),
            iterations=1,
        )
        refined = cv2.dilate(
            refined,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)),
            iterations=1,
        )

        contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_sub_area = max(80, min_area // 3)
        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_sub_area:
                continue

            sx, sy, sw, sh = cv2.boundingRect(contour)
            if sw < 8 or sh < 8:
                continue

            x1 = max(0, rx + sx - padding)
            y1 = max(0, ry + sy - padding)
            x2 = min(page_w, rx + sx + sw + padding)
            y2 = min(page_h, ry + sy + sh + padding)
            boxes.append((x1, y1, x2, y2))

        if not boxes:
            return []

        merge_gap = max(8, int(short_side * 0.05))
        return _merge_nearby_boxes(boxes, gap=merge_gap)

    def _make_region(
        self,
        image: NDArray,
        mask: NDArray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> TextRegion:
        """Create a TextRegion object from absolute coordinates."""
        region_mask = mask[y1:y2, x1:x2].copy()
        cropped = image[y1:y2, x1:x2].copy()
        return TextRegion(
            x=x1,
            y=y1,
            w=x2 - x1,
            h=y2 - y1,
            cropped=cropped,
            mask=region_mask,
        )

    def _split_disconnected_region(
        self,
        mask: NDArray,
        region_bbox: tuple[int, int, int, int],
        min_area: int,
        padding: int,
        page_w: int,
        page_h: int,
    ) -> list[tuple[int, int, int, int]]:
        """
        Split medium/large regions that contain clearly disconnected text clusters.
        """
        rx, ry, rw, rh = region_bbox
        sub_mask = mask[ry : ry + rh, rx : rx + rw]
        if cv2.countNonZero(sub_mask) == 0:
            return []

        short_side = max(1, min(rw, rh))
        close_size = _odd(max(3, int(round(short_side * 0.012))))
        dilate_size = _odd(max(5, int(round(short_side * 0.03))))

        refined = cv2.morphologyEx(
            sub_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size)),
            iterations=1,
        )
        refined = cv2.dilate(
            refined,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)),
            iterations=1,
        )

        contours, _ = cv2.findContours(refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_sub_area = max(60, min_area // 3)

        boxes: list[tuple[int, int, int, int]] = []
        areas: list[float] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_sub_area:
                continue
            sx, sy, sw, sh = cv2.boundingRect(contour)
            if sw < 8 or sh < 8:
                continue

            x1 = max(0, rx + sx - padding)
            y1 = max(0, ry + sy - padding)
            x2 = min(page_w, rx + sx + sw + padding)
            y2 = min(page_h, ry + sy + sh + padding)
            boxes.append((x1, y1, x2, y2))
            areas.append(area)

        if len(boxes) < 2:
            return []

        merged = _merge_nearby_boxes(boxes, gap=max(10, int(short_side * 0.10)))
        if len(merged) < 2:
            return []

        total_area = float(sum(areas)) if areas else 0.0
        max_area = float(max(areas)) if areas else 0.0
        if total_area > 0.0 and (max_area / total_area) > 0.82:
            return []

        separation_threshold = max(20, int(short_side * 0.18))
        separated = False
        for i in range(len(merged)):
            ax1, ay1, ax2, ay2 = merged[i]
            for j in range(i + 1, len(merged)):
                bx1, by1, bx2, by2 = merged[j]
                x_gap = max(0, max(ax1, bx1) - min(ax2, bx2))
                y_gap = max(0, max(ay1, by1) - min(ay2, by2))
                if x_gap > separation_threshold or y_gap > separation_threshold:
                    separated = True
                    break
            if separated:
                break

        return merged if separated else []

    def _is_region_viable(self, region: TextRegion) -> bool:
        """Filter noisy tiny detections that are unlikely to be useful text regions."""
        if region.w < 8 or region.h < 8:
            return False
        if region.mask is None:
            return True

        non_zero = cv2.countNonZero(region.mask.astype(np.uint8))
        if non_zero < 10:
            return False
        return True


class ModalTextDetector(ComicTextDetector):
    """
    Detector that offloads ONNX mask inference to a Modal CPU service.

    Region extraction and sorting still run locally so downstream behavior
    stays aligned with the existing pipeline.
    """

    _instance: ModalTextDetector | None = None
    _modal_detector = None
    _modal_lock = Lock()

    def __new__(cls) -> ModalTextDetector:
        if cls._instance is None:
            cls._instance = super(ComicTextDetector, cls).__new__(cls)
        return cls._instance

    def _ensure_modal(self) -> None:
        if self._modal_detector is not None:
            return
        with self._modal_lock:
            if self._modal_detector is not None:
                return
            import modal

            logger.info("Connecting to Modal comic-text-detector service...")
            DetectorCls = modal.Cls.from_name(_MODAL_DETECTOR_APP, _MODAL_DETECTOR_CLASS)
            self._modal_detector = DetectorCls()
            logger.info("Connected to Modal comic-text-detector service.")

    @staticmethod
    def _encode_png(image: NDArray) -> bytes:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Failed to encode image for Modal detector call.")
        return encoded.tobytes()

    @staticmethod
    def _decode_mask(mask_bytes: bytes, expected_h: int, expected_w: int) -> NDArray:
        if not mask_bytes:
            return np.zeros((expected_h, expected_w), dtype=np.uint8)

        payload = np.frombuffer(mask_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(payload, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            raise ValueError("Modal detector returned invalid mask bytes.")

        if decoded.shape != (expected_h, expected_w):
            decoded = cv2.resize(
                decoded,
                (expected_w, expected_h),
                interpolation=cv2.INTER_NEAREST,
            )
        return (decoded > 0).astype(np.uint8) * 255

    def detect(
        self,
        image: NDArray,
        text_threshold: float = 0.25,
        min_area: int = 30,
        padding: int = 5,
    ) -> tuple[list[TextRegion], NDArray]:
        self._ensure_modal()

        h, w = image.shape[:2]
        image_bytes = self._encode_png(image)

        try:
            text_mask_bytes = self._modal_detector.detect_mask.remote(
                image_bytes, text_threshold
            )
        except Exception as e:
            raise RuntimeError(
                "Modal detection failed. Deploy `modal_detector_service.py` and "
                "ensure your Modal auth/env vars are configured."
            ) from e

        text_mask = self._decode_mask(text_mask_bytes, expected_h=h, expected_w=w)
        regions = self._extract_regions_from_mask(
            image, text_mask, min_area=min_area, padding=padding
        )
        regions = _sort_manga_order(regions, page_width=w)
        logger.info("Detected %d text regions using Modal comic-text-detector.", len(regions))
        return regions, text_mask

    def _ensure_model(self) -> None:
        # No local ONNX session required when using Modal detector service.
        pass


def get_text_detector() -> ComicTextDetector:
    """Factory for selecting local vs Modal detection backend."""
    from .config import settings

    if settings.use_detection_model:
        logger.info("Using Modal comic-text-detector service.")
        return ModalTextDetector()

    logger.info("Using local comic-text-detector model.")
    return ComicTextDetector()


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
    detector = get_text_detector()
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
