"""Backend-facing inference service for the trained CNN-LSTM model."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from config import PROJECT_ROOT

ML_DIR = PROJECT_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from inference import ModelSingleton, _prepare_images, _prepare_sensors, predict_stress  # noqa: E402
from utils import CLASS_LABELS, DEFAULT_SEQUENCE_LENGTH, REPORTS_DIR, SENSOR_COLUMNS  # noqa: E402


class InferenceService:
    """Wrap real model predictions and derived explainability summaries."""

    _activation_model = None
    _activation_lock = Lock()

    @staticmethod
    def _default_sensor_reading() -> dict[str, float]:
        stats_path = REPORTS_DIR / "preprocessing_statistics.json"
        normalization = json.loads(stats_path.read_text())["sensor_normalization"]
        return {
            column: float(normalization["mean"][index])
            for index, column in enumerate(SENSOR_COLUMNS)
        }

    @staticmethod
    def normalize_sensor_readings(readings: Any) -> tuple[list[dict[str, float]], dict[str, Any]]:
        """Accept dict/list sensor input and pad or trim to the model's sequence length."""
        parsed: list[dict[str, float]] = []
        readings = readings or []
        if not isinstance(readings, list):
            raise ValueError("recent_sensor_readings must be a list.")

        for index, row in enumerate(readings):
            if isinstance(row, dict):
                parsed_row = {}
                for column in SENSOR_COLUMNS:
                    if column not in row:
                        raise ValueError(f"Reading {index + 1} is missing {column}.")
                    parsed_row[column] = float(row[column])
            elif isinstance(row, (list, tuple)) and len(row) == len(SENSOR_COLUMNS):
                parsed_row = {
                    column: float(row[column_index])
                    for column_index, column in enumerate(SENSOR_COLUMNS)
                }
            else:
                raise ValueError(
                    "Each sensor reading must be an object with Soil_Moisture, "
                    "Temperature, Humidity, Light_Intensity, or a 4-value list."
                )
            parsed.append(parsed_row)

        adjustment = {
            "requested_length": len(parsed),
            "required_length": DEFAULT_SEQUENCE_LENGTH,
            "strategy": "unchanged",
            "message": "Received exactly the required number of readings.",
        }

        if not parsed:
            default = InferenceService._default_sensor_reading()
            parsed = [default.copy() for _ in range(DEFAULT_SEQUENCE_LENGTH)]
            adjustment.update({
                "strategy": "default_mean_sequence",
                "message": "No readings supplied; used training-set mean sensor values.",
            })
        elif len(parsed) < DEFAULT_SEQUENCE_LENGTH:
            last = parsed[-1].copy()
            parsed.extend(last.copy() for _ in range(DEFAULT_SEQUENCE_LENGTH - len(parsed)))
            adjustment.update({
                "strategy": "padded_repeat_last",
                "message": "Fewer than 7 readings supplied; repeated the most recent reading.",
            })
        elif len(parsed) > DEFAULT_SEQUENCE_LENGTH:
            parsed = parsed[-DEFAULT_SEQUENCE_LENGTH:]
            adjustment.update({
                "strategy": "trimmed_to_latest",
                "message": "More than 7 readings supplied; used the latest 7 readings.",
            })

        return parsed, adjustment

    @staticmethod
    def _sensor_matrix(readings: list[dict[str, float]]) -> np.ndarray:
        return np.asarray([[row[column] for column in SENSOR_COLUMNS] for row in readings], dtype="float32")

    @classmethod
    def _get_activation_model(cls):
        if cls._activation_model is None:
            with cls._activation_lock:
                if cls._activation_model is None:
                    tf = __import__("tensorflow")
                    model = ModelSingleton.get()
                    cls._activation_model = tf.keras.Model(
                        inputs=model.inputs,
                        outputs=model.get_layer("td_conv_128").output,
                        name="agrisense_activation_probe",
                    )
        return cls._activation_model

    @staticmethod
    def _top_regions(heatmap: np.ndarray, grid_size: int = 4) -> list[dict[str, float | str]]:
        rows, columns = heatmap.shape
        regions: list[dict[str, float | str]] = []
        for row in range(grid_size):
            for column in range(grid_size):
                y0 = int(row * rows / grid_size)
                y1 = int((row + 1) * rows / grid_size)
                x0 = int(column * columns / grid_size)
                x1 = int((column + 1) * columns / grid_size)
                score = float(heatmap[y0:y1, x0:x1].mean())
                regions.append({
                    "region": f"r{row + 1}c{column + 1}",
                    "row": row + 1,
                    "column": column + 1,
                    "score": round(score, 6),
                })
        return sorted(regions, key=lambda item: item["score"], reverse=True)[:3]

    @staticmethod
    def _downsample_heatmap(heatmap: np.ndarray, grid_size: int = 7) -> list[list[float]]:
        rows, columns = heatmap.shape
        grid: list[list[float]] = []
        for row in range(grid_size):
            values: list[float] = []
            for column in range(grid_size):
                y0 = int(row * rows / grid_size)
                y1 = int((row + 1) * rows / grid_size)
                x0 = int(column * columns / grid_size)
                x1 = int((column + 1) * columns / grid_size)
                values.append(round(float(heatmap[y0:y1, x0:x1].mean()), 6))
            grid.append(values)
        return grid

    @classmethod
    def cnn_feature_summary(cls, image_path: str, sensor_matrix: np.ndarray) -> dict[str, Any]:
        """Compute a compact real activation summary from the last CNN block."""
        images = _prepare_images([image_path] * DEFAULT_SEQUENCE_LENGTH)[None, ...]
        sensors = _prepare_sensors(sensor_matrix)[None, ...]
        activations = cls._get_activation_model().predict(
            {"image_sequence": images, "sensor_sequence": sensors}, verbose=0
        )
        last_step = np.asarray(activations[0, -1], dtype="float32")
        heatmap = last_step.mean(axis=-1)
        maximum = float(heatmap.max())
        normalized = heatmap / maximum if maximum > 0 else heatmap
        return {
            "source_layer": "td_conv_128",
            "activation_energy": round(float(normalized.mean()), 6),
            "peak_activation": round(float(normalized.max()), 6),
            "top_regions": cls._top_regions(normalized),
            "heatmap_grid": cls._downsample_heatmap(normalized),
            "heatmap_shape": list(normalized.shape),
        }

    @staticmethod
    def lstm_trend(readings: list[dict[str, float]]) -> dict[str, Any]:
        matrix = InferenceService._sensor_matrix(readings)
        soil = matrix[:, 0]
        temperature = matrix[:, 1]
        humidity = matrix[:, 2]
        # A simple stress proxy: higher when soil/humidity fall and temperature rises.
        proxy = (-soil / 100.0) + (temperature / 45.0) + (-humidity / 100.0)
        x_axis = np.arange(len(proxy), dtype="float32")
        slope = float(np.polyfit(x_axis, proxy, deg=1)[0]) if len(proxy) > 1 else 0.0
        if slope > 0.01:
            direction = "declining"
        elif slope < -0.01:
            direction = "improving"
        else:
            direction = "stable"
        return {
            "direction": direction,
            "slope": round(slope, 6),
            "stress_proxy": [round(float(value), 6) for value in proxy],
        }

    @classmethod
    def predict(cls, image_path: str, sensor_readings: list[dict[str, float]]) -> dict[str, Any]:
        started = time.perf_counter()
        sensor_matrix = cls._sensor_matrix(sensor_readings)
        result = predict_stress([image_path] * DEFAULT_SEQUENCE_LENGTH, sensor_matrix)
        feature_summary = cls.cnn_feature_summary(image_path, sensor_matrix)
        trend = cls.lstm_trend(sensor_readings)
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "predicted_class": result["predicted_class"],
            "confidence": result["confidence"],
            "class_probabilities": result["class_probabilities"],
            "cnn_feature_summary": feature_summary,
            "lstm_trend": trend,
            "prediction_time_ms": total_ms,
            "timestamp": result["timestamp"],
            "class_labels": CLASS_LABELS,
        }
