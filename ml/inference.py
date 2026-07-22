"""Singleton inference wrapper for real AgriSense softmax predictions."""

from __future__ import annotations

import time
from pathlib import Path
from threading import Lock

import numpy as np
from PIL import Image

from artifacts import load_contract
from utils import CLASS_LABELS, DEFAULT_SEQUENCE_LENGTH, MODEL_PATH, SENSOR_COLUMNS, require_tensorflow, utc_now


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
                    contract = load_contract()
                    tf = require_tensorflow()
                    cls._model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                    input_names = [tensor.name.split(":", 1)[0] for tensor in cls._model.inputs]
                    if input_names != contract["input_names"]:
                        raise ValueError(
                            f"Model inputs {input_names} do not match contract {contract['input_names']}"
                        )
        return cls._model


def _prepare_image(item) -> np.ndarray:
    contract = load_contract()
    height, width, channels = contract["image_size"]
    if channels != 3:
        raise ValueError(f"Unsupported image channel count in contract: {channels}")
    if isinstance(item, (str, Path)):
        with Image.open(item) as image:
            array = np.asarray(image.convert("RGB").resize((width, height)), dtype="float32")
    else:
        array = np.asarray(item)
        if array.shape != (height, width, channels):
            pil_values = array
            if np.issubdtype(pil_values.dtype, np.floating) and pil_values.size:
                if np.nanmax(pil_values) <= 1.0:
                    pil_values = pil_values * 255.0
            with Image.fromarray(np.asarray(pil_values, dtype="uint8")) as image:
                array = np.asarray(
                    image.convert("RGB").resize((width, height)), dtype="float32"
                )
        else:
            array = array.astype("float32")
    if not np.isfinite(array).all():
        raise ValueError("Image contains non-finite values")
    if array.max() > 1.0:
        array /= 255.0
    return array.astype("float32")


def _prepare_sensors(sensor_sequence) -> np.ndarray:
    values = np.asarray(sensor_sequence, dtype="float32")
    contract = load_contract()
    expected_shape = (contract["sequence_length"], len(contract["sensor_columns"]))
    if values.shape != expected_shape:
        raise ValueError(f"Expected sensor shape {expected_shape}, received {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Sensor sequence contains non-finite values")
    mean = np.asarray(contract["sensor_mean"], dtype="float32")
    std = np.asarray(contract["sensor_std"], dtype="float32")
    return (values - mean) / std


def predict_stress(image, sensor_sequence) -> dict:
    """Predict stress from one current image and seven sensor readings."""
    started = time.perf_counter()
    contract = load_contract()
    images = _prepare_image(image)[None, ...]
    sensors = _prepare_sensors(sensor_sequence)[None, ...]
    raw = ModelSingleton.get().predict(
        {"image": images, "sensor_sequence": sensors}, verbose=0
    )
    if isinstance(raw, dict):
        probabilities = np.asarray(raw["stress_probabilities"][0], dtype="float32")
        crop_values = np.asarray(raw["crop_probabilities"][0], dtype="float32")
    else:
        probabilities = np.asarray(raw[0], dtype="float32")
        crop_values = np.asarray([], dtype="float32")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Model returned invalid class probabilities")
    if not np.isclose(float(probabilities.sum()), 1.0, atol=1e-5):
        raise ValueError("Model class probabilities do not sum to one")
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
        "probability_array": probabilities.tolist(),
        "crop_probabilities": {
            label: float(crop_values[i])
            for i, label in enumerate(contract.get("crop_labels", []))
            if i < len(crop_values)
        },
        "timestamp": utc_now(),
        "latency_ms": latency_ms,
        "prediction_time_ms": latency_ms,
    }
