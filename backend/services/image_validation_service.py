"""Content and quality gate for uploaded leaf images."""

from __future__ import annotations

import sys
from pathlib import Path
from threading import Lock
from typing import Callable

import numpy as np
from PIL import Image, ImageOps

from config import PROJECT_ROOT

ML_DIR = PROJECT_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from artifacts import load_contract  # noqa: E402
from calibration import leaf_similarity, quality_metrics  # noqa: E402
from inference import ModelSingleton, _prepare_image  # noqa: E402
from utils import require_tensorflow  # noqa: E402


class ImageValidationError(ValueError):
    def __init__(self, result: dict):
        super().__init__(result["message"])
        self.result = result


class ImageValidationService:
    _embedding_model = None
    _lock = Lock()

    def __init__(
        self,
        contract: dict | None = None,
        embedding_provider: Callable[[Image.Image], np.ndarray] | None = None,
    ) -> None:
        self.contract = contract or load_contract()
        self.reference = self.contract.get("leaf_validation")
        self.embedding_provider = embedding_provider or self._model_embedding

    @classmethod
    def _get_embedding_model(cls):
        if cls._embedding_model is None:
            with cls._lock:
                if cls._embedding_model is None:
                    tf = require_tensorflow()
                    model = ModelSingleton.get()
                    image_input = tf.keras.Input(shape=(128, 128, 3))
                    scaled = model.get_layer("imagenet_rescaling")(image_input)
                    feature_map = model.get_layer("image_encoder")(scaled, training=False)
                    pooled = tf.keras.layers.GlobalAveragePooling2D()(feature_map)
                    cls._embedding_model = tf.keras.Model(image_input, pooled)
        return cls._embedding_model

    @classmethod
    def _model_embedding(cls, image: Image.Image) -> np.ndarray:
        array = _prepare_image(np.asarray(image))[None, ...]
        return np.asarray(cls._get_embedding_model().predict(array, verbose=0)[0])

    @staticmethod
    def _load_rgb(image_or_path) -> Image.Image:
        if isinstance(image_or_path, Image.Image):
            return ImageOps.exif_transpose(image_or_path).convert("RGB")
        with Image.open(Path(image_or_path)) as image:
            return ImageOps.exif_transpose(image).convert("RGB")

    @staticmethod
    def _result(status: str, code: str, message: str, guidance: list[str], **extra) -> dict:
        return {
            "status": status,
            "reason_code": code,
            "message": message,
            "guidance": guidance,
            **extra,
        }

    def validate(self, image_or_path) -> dict:
        image = self._load_rgb(image_or_path)
        metrics = quality_metrics(np.asarray(image))
        if image.width < 96 or image.height < 96:
            return self._result(
                "retry_required", "too_small", "The image is too small for reliable analysis.",
                ["Use an image at least 96 × 96 pixels."], quality=metrics,
            )
        reference = self.reference
        if reference is None:
            return self._result(
                "retry_required", "validation_unavailable",
                "The deployed model does not include image-validation calibration.",
                ["Retrain and promote a schema-v2 model before analyzing uploads."],
                quality=metrics,
            )
        quality = reference["quality"]
        if metrics["contrast"] < quality["min_contrast"] and metrics["sharpness"] < quality["min_sharpness"]:
            return self._result(
                "rejected", "near_uniform", "The image has no usable leaf detail.",
                ["Upload a clear photo containing one visible leaf."], quality=metrics,
            )
        if metrics["brightness"] < quality["min_brightness"] or metrics["dark_clip"] > quality["max_dark_clip"]:
            return self._result(
                "retry_required", "too_dark", "The image is too dark for reliable analysis.",
                ["Retake the photo in even natural light."], quality=metrics,
            )
        if metrics["brightness"] > quality["max_brightness"] or metrics["bright_clip"] > quality["max_bright_clip"]:
            return self._result(
                "retry_required", "overexposed", "The image is overexposed.",
                ["Avoid glare and reduce direct light."], quality=metrics,
            )
        if metrics["sharpness"] < quality["min_sharpness"]:
            return self._result(
                "retry_required", "blurry", "The image is too blurry for reliable analysis.",
                ["Hold the camera steady and focus on the leaf surface."], quality=metrics,
            )
        score, crop = leaf_similarity(self.embedding_provider(image), reference)
        if score < reference["retry_threshold"]:
            status, code = "rejected", "non_leaf"
            message = "This image does not look like a supported leaf."
        elif score < reference["accept_threshold"]:
            status, code = "retry_required", "uncertain_leaf"
            message = "The leaf content is uncertain; please retake the image."
        else:
            status, code = "accepted", "valid_leaf"
            message = "Leaf image accepted for analysis."
        return self._result(
            status, code, message,
            ["Upload one clear leaf centered in the frame."] if status != "accepted" else [],
            leaf_similarity=round(score, 6), inferred_crop=crop, quality=metrics,
            threshold_version=reference["version"],
        )
