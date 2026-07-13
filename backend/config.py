"""Configuration for the AgriSense AI Flask API."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
UPLOAD_DIR = BACKEND_DIR / "static" / "uploads"
DATABASE_PATH = BACKEND_DIR / "agrisense.sqlite3"
DATASET_CSV = PROJECT_ROOT / "dataset" / "sequential_data.csv"
MODEL_METADATA_PATH = PROJECT_ROOT / "ml" / "reports" / "model_metadata.json"
ML_REPORTS_DIR = PROJECT_ROOT / "ml" / "reports"
MODEL_PATH = PROJECT_ROOT / "ml" / "saved_model" / "agrisense_cnn_lstm.keras"


class Config:
    SECRET_KEY = os.getenv("AGRISENSE_SECRET_KEY", "dev-only-agrisense-secret")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", f"sqlite:///{DATABASE_PATH}")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_UPLOAD_MB", "8")) * 1024 * 1024
    JSON_SORT_KEYS = False
    CORS_ORIGINS = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    ALLOWED_SENSOR_EXTENSIONS = {"csv"}


def ensure_backend_directories() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
