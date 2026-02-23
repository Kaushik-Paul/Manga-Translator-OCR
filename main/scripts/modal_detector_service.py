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
