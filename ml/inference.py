"""Singleton inference wrapper for real AgriSense softmax predictions."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

from model import macro_f1_metric
from utils import CLASS_LABELS, DEFAULT_SEQUENCE_LENGTH, MODEL_PATH, REPORTS_DIR, SENSOR_COLUMNS, require_tensorflow, utc_now


class ModelSingleton:
    _model = None
    _lock = Lock()

    @classmethod
    def get(cls):
        if cls._model is None:
            with cls._lock:
                if cls._model is None:
                    if not MODEL_PATH.is_file():
                        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")
                    tf = require_tensorflow()
                    cls._model = tf.keras.models.load_model(
                        MODEL_PATH, custom_objects={"MacroF1": type(macro_f1_metric())}
                    )
        return cls._model


def _prepare_images(image_sequence) -> np.ndarray:
    if len(image_sequence) != DEFAULT_SEQUENCE_LENGTH:
        raise ValueError(f"Expected {DEFAULT_SEQUENCE_LENGTH} images, received {len(image_sequence)}")
    images = []
    for item in image_sequence:
        if isinstance(item, (str, Path)):
            with Image.open(item) as image:
                array = np.asarray(image.convert("RGB").resize((128, 128)), dtype="float32")
        else:
            array = np.asarray(item, dtype="float32")
            if array.shape != (128, 128, 3):
                with Image.fromarray(array.astype("uint8")) as image:
                    array = np.asarray(image.convert("RGB").resize((128, 128)), dtype="float32")
        if array.max() > 1.0:
            array /= 255.0
        images.append(array)
    return np.asarray(images, dtype="float32")


def _prepare_sensors(sensor_sequence) -> np.ndarray:
    values = np.asarray(sensor_sequence, dtype="float32")
    if values.shape != (DEFAULT_SEQUENCE_LENGTH, len(SENSOR_COLUMNS)):
        raise ValueError(f"Expected sensor shape ({DEFAULT_SEQUENCE_LENGTH}, 4), received {values.shape}")
    import json
    stats_path = REPORTS_DIR / "preprocessing_statistics.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing preprocessing statistics: {stats_path}")
    normalization = json.loads(stats_path.read_text())["sensor_normalization"]
    mean = np.asarray(normalization["mean"], dtype="float32")
    std = np.asarray(normalization["std"], dtype="float32")
    return (values - mean) / std


def predict_stress(image_sequence, sensor_sequence) -> dict:
    """Predict stress from seven images/readings and return JSON-compatible values."""
    started = time.perf_counter()
    images = _prepare_images(image_sequence)[None, ...]
    sensors = _prepare_sensors(sensor_sequence)[None, ...]
    probabilities = ModelSingleton.get().predict(
        {"image_sequence": images, "sensor_sequence": sensors}, verbose=0
    )[0]
    index = int(np.argmax(probabilities))
    predicted_class = CLASS_LABELS[index]
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "class": predicted_class,
        "predicted_class": predicted_class,
        "confidence": float(probabilities[index]),
        "class_probabilities": {
            label: float(probabilities[i]) for i, label in enumerate(CLASS_LABELS)
        },
        "timestamp": utc_now(),
        "latency_ms": latency_ms,
        "prediction_time_ms": latency_ms,
    }
