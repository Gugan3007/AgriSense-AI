"""Smoke test for the AgriSense Flask API.

Run from the project root:
    .venv/bin/python backend/tests/smoke_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import create_app  # noqa: E402
from models.db_models import db  # noqa: E402


def assert_status(response, expected: int, label: str) -> dict:
    if response.status_code != expected:
        raise AssertionError(
            f"{label} expected HTTP {expected}, got {response.status_code}: "
            f"{response.get_data(as_text=True)}"
        )
    return response.get_json()


def main() -> None:
    app = create_app()
    client = app.test_client()

    with app.app_context():
        db.create_all()

    dataset_info = assert_status(client.get("/dataset-info"), 200, "dataset-info")
    assert dataset_info["plant_count"] >= 60

    model_info = assert_status(client.get("/model-info"), 200, "model-info")
    assert "best_validation_accuracy" in model_info

    frame = pd.read_csv(PROJECT_ROOT / "dataset" / "sequential_data.csv")
    sample = frame.sort_values(["Plant_ID", "Timestamp"]).head(7)
    image_path = PROJECT_ROOT / sample.iloc[-1]["Image_Path"]
    readings = sample[["Soil_Moisture", "Temperature", "Humidity", "Light_Intensity"]].to_dict(
        orient="records"
    )

    with image_path.open("rb") as handle:
        upload_response = client.post(
            "/upload",
            data={"image": (handle, image_path.name)},
            content_type="multipart/form-data",
        )
    upload = assert_status(upload_response, 201, "upload")

    prediction = assert_status(
        client.post(
            "/predict",
            json={
                "upload_id": upload["upload_id"],
                "plant_type": sample.iloc[-1]["Plant_Type"],
                "recent_sensor_readings": readings,
            },
        ),
        201,
        "predict",
    )
    assert prediction["predicted_class"] in {"Healthy", "Low", "Medium", "High"}
    assert 0 <= prediction["confidence"] <= 1
    assert prediction["cnn_feature_summary"]["top_regions"]

    history = assert_status(client.get("/history?page=1&limit=5"), 200, "history")
    assert history["total"] >= 1

    detail = assert_status(client.get(f"/history/{prediction['prediction_id']}"), 200, "history-detail")
    assert detail["prediction_id"] == prediction["prediction_id"]

    contact = assert_status(
        client.post(
            "/contact",
            json={
                "name": "AgriSense Reviewer",
                "email": "reviewer@example.com",
                "message": "Smoke test contact message from the backend verification script.",
            },
        ),
        201,
        "contact",
    )
    assert contact["success"] is True

    print(json.dumps({
        "status": "ok",
        "dataset_rows": dataset_info["row_count"],
        "best_validation_accuracy": model_info["best_validation_accuracy"],
        "upload_id": upload["upload_id"],
        "prediction_id": prediction["prediction_id"],
        "predicted_class": prediction["predicted_class"],
        "confidence": prediction["confidence"],
        "history_total": history["total"],
    }, indent=2))


if __name__ == "__main__":
    main()
