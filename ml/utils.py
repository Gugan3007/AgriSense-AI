"""Shared constants and filesystem helpers for the AgriSense ML layer."""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "ml"
DATASET_CSV = PROJECT_ROOT / "dataset" / "sequential_data.csv"
REPORTS_DIR = ML_DIR / "reports"
SAVED_MODEL_DIR = ML_DIR / "saved_model"
TENSORBOARD_DIR = ML_DIR / "tensorboard_logs"
MODEL_PATH = SAVED_MODEL_DIR / "agrisense_cnn_lstm.keras"
CLASS_LABELS = ["Healthy", "Low", "Medium", "High"]
SENSOR_COLUMNS = ["Soil_Moisture", "Temperature", "Humidity", "Light_Intensity"]
IMAGE_SIZE = (128, 128)
DEFAULT_SEQUENCE_LENGTH = 7
SEED = 42


def ensure_directories() -> None:
    for path in (REPORTS_DIR, SAVED_MODEL_DIR, TENSORBOARD_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_reproducible_seed(seed: int = SEED) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required. Install Phase 2 dependencies with: "
            "python3 -m pip install -r ml/requirements.txt"
        ) from exc
    return tf

