"""
Modal service for comic-text-detector ONNX inference.

Deploy with:
    modal deploy modal_detector_service.py

This creates a remotely-hosted CPU service that runs comic-text-detector.
The local client calls it via:
`modal.Cls.from_name("manga-detector-service", "ComicTextDetectorService")`.
"""

from __future__ import annotations

import modal

_MODEL_REPO = "mayocream/comic-text-detector-onnx"
_MODEL_FILE = "comic-text-detector.onnx"
_MODEL_INPUT_SIZE = (1024, 1024)
_BLOCK_CONF_THRESHOLD = 0.4
_BLOCK_NMS_THRESHOLD = 0.35

app = modal.App("manga-detector-service")

detector_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "huggingface_hub>=0.26",
        "numpy>=1.26",
        "onnxruntime>=1.17",
        "opencv-python-headless>=4.9",
    )
)

_MODAL_SECRETS = [modal.Secret.from_name("huggingface-secret")]


@app.cls(
    image=detector_image,
    secrets=_MODAL_SECRETS,
    cpu=1,
    memory=2048,
    scaledown_window=120,
    timeout=300,
)
class ComicTextDetectorService:
    """
    Runs comic-text-detector ONNX inference on Modal CPU.

    The model is loaded once per container start via @modal.enter().
    """

    @modal.enter()
    def load_model(self) -> None:
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort

        model_path = hf_hub_download(
            repo_id=_MODEL_REPO,
            filename=_MODEL_FILE,
        )

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    @modal.method()
    def detect_mask(self, image_bytes: bytes, threshold: float = 0.5) -> bytes:
        """
        Run detection and return a binary text mask as PNG bytes.

        Args:
            image_bytes: Raw PNG/JPEG bytes of the page image.
            threshold: Binarization threshold in [0, 1].

        Returns:
            PNG-encoded uint8 mask with shape matching input image.
        """
        import cv2
        import numpy as np

        threshold = max(0.0, min(1.0, float(threshold)))

        payload = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
        if image is None:
            return b""

        orig_h, orig_w = image.shape[:2]
        input_tensor = self._preprocess(image)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})

        mask = self._parse_mask(outputs, orig_h=orig_h, orig_w=orig_w, threshold=threshold)
        ok, encoded = cv2.imencode(".png", mask)
        if not ok:
            return b""
        return encoded.tobytes()

    @modal.method()
    def detect_layout(self, image_bytes: bytes, threshold: float = 0.5) -> dict:
        """
        Run detection with letterboxed input and return the text mask plus
        the YOLO text-block boxes.

        Returns:
            {"mask": PNG bytes of the binary mask at page size,
             "blocks": [[x1, y1, x2, y2, score], ...] in page coordinates}
        """
        import cv2
        import numpy as np

        threshold = max(0.0, min(1.0, float(threshold)))
        payload = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(payload, cv2.IMREAD_COLOR)
        if image is None:
            return {"mask": b"", "blocks": []}

        orig_h, orig_w = image.shape[:2]
        in_w, in_h = _MODEL_INPUT_SIZE
        ratio = min(in_w / float(orig_w), in_h / float(orig_h))
        new_w = max(1, min(in_w, int(round(orig_w * ratio))))
        new_h = max(1, min(in_h, int(round(orig_h * ratio))))
        interp = cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR
        canvas = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(image, (new_w, new_h), interpolation=interp)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]

        names = [o.name for o in self.session.get_outputs()]
        outputs = dict(zip(names, self.session.run(None, {self.session.get_inputs()[0].name: blob})))

        seg = np.asarray(outputs.get("seg"), dtype=np.float32).squeeze()
        if seg.ndim == 3:
            seg = seg[0]
        if seg.ndim == 2 and seg.size:
            if float(seg.max()) > 1.0:
                seg = seg / float(seg.max())
            seg = cv2.resize(seg[:new_h, :new_w], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            mask = (seg > threshold).astype(np.uint8) * 255
        else:
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", mask)
        mask_bytes = encoded.tobytes() if ok else b""

        blocks: list[list[float]] = []
        blk = outputs.get("blk")
        if blk is not None:
            arr = np.asarray(blk, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[0]
            if arr.ndim == 2 and arr.shape[1] >= 6 and arr.shape[0]:
                scores = arr[:, 4] * arr[:, 5:].max(axis=1)
                keep = scores > _BLOCK_CONF_THRESHOLD
                arr, scores = arr[keep], scores[keep]
                rects = [
                    [float(cx - bw / 2.0), float(cy - bh / 2.0), float(bw), float(bh)]
                    for cx, cy, bw, bh in arr[:, :4]
                ]
                picked = (
                    cv2.dnn.NMSBoxes(rects, scores.tolist(), _BLOCK_CONF_THRESHOLD, _BLOCK_NMS_THRESHOLD)
                    if rects
                    else []
                )
                scale = 1.0 / max(ratio, 1e-6)
                for idx in np.asarray(picked).reshape(-1):
                    x, y, bw, bh = rects[int(idx)]
                    x1 = int(np.clip(round(x * scale), 0, orig_w))
                    y1 = int(np.clip(round(y * scale), 0, orig_h))
                    x2 = int(np.clip(round((x + bw) * scale), 0, orig_w))
                    y2 = int(np.clip(round((y + bh) * scale), 0, orig_h))
                    if x2 - x1 >= 4 and y2 - y1 >= 4:
                        blocks.append([x1, y1, x2, y2, float(scores[int(idx)])])

        return {"mask": mask_bytes, "blocks": blocks}

    def _preprocess(self, image):
        import cv2
        import numpy as np

        resized = cv2.resize(image, _MODEL_INPUT_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)
        return blob

    def _parse_mask(self, outputs, orig_h: int, orig_w: int, threshold: float):
        import cv2
        import numpy as np

        mask = None
        for out in outputs:
            arr = np.array(out)
            if len(arr.shape) < 2:
                continue

            flat = arr.squeeze()
            if len(flat.shape) < 2:
                continue

            spatial_dims = flat.shape[-2:]
            if spatial_dims[0] < 256 or spatial_dims[1] < 256:
                continue

            if len(flat.shape) == 3:
                flat = flat[0] if flat.shape[0] <= 4 else flat[-1]
            mask = flat
            break

        if mask is None:
            best_size = 0
            for out in outputs:
                arr = np.array(out).squeeze()
                if len(arr.shape) < 2:
                    continue
                size = arr.shape[-1] * arr.shape[-2]
                if size <= best_size:
                    continue
                best_size = size
                mask = arr if len(arr.shape) == 2 else arr[0]

        if mask is None:
            return np.zeros((orig_h, orig_w), dtype=np.uint8)

        mask = np.array(mask, dtype=np.float32)
        max_val = float(mask.max()) if mask.size > 0 else 0.0
        if max_val > 1.0:
            mask = mask / max_val

        resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary = (resized > threshold).astype(np.uint8) * 255
        return binary
