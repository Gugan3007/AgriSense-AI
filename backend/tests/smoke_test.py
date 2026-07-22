"""Smoke test for the AgriSense Flask API.

Run from the project root:
    .venv/bin/python backend/tests/smoke_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
ML_DIR = PROJECT_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from app import create_app  # noqa: E402
from artifacts import load_contract  # noqa: E402
from calibration import classify_reliability  # noqa: E402
from models.db_models import db  # noqa: E402
from preprocessing import (  # noqa: E402
    CROP_LABELS, SENSOR_COLUMNS, fit_sensor_normalization, load_valid_rows,
    make_windows, split_plant_ids,
)
from services.image_validation_service import ImageValidationService  # noqa: E402


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

    frame = load_valid_rows()
    splits = split_plant_ids(frame)
    normalization = fit_sensor_normalization(frame, splits["train"])
    paths, sensors, _, crops = make_windows(frame, splits["test"], 7, normalization)
    report = pd.read_csv(PROJECT_ROOT / "ml" / "reports" / "test_predictions.csv")
    contract = load_contract()
    probability_columns = [f"probability_{label}" for label in ("Healthy", "Low", "Medium", "High")]
    assert len(report) == len(paths)

    validator = ImageValidationService()
    selected = {}
    for index, row in report.iterrows():
        status = classify_reliability(
            row[probability_columns].to_numpy(dtype="float32"),
            contract["decision_policy"],
        )["analysis_status"]
        if status not in selected and validator.validate(paths[index])["status"] == "accepted":
            selected[status] = index
        if set(selected) == {"completed", "inconclusive"}:
            break
    assert set(selected) == {"completed", "inconclusive"}

    predictions = {}
    mean = np.asarray(normalization["mean"], dtype="float32")
    std = np.asarray(normalization["std"], dtype="float32")
    for expected_status, index in selected.items():
        image_path = Path(paths[index])
        raw_readings = sensors[index] * std + mean
        readings = [
            {column: float(value) for column, value in zip(SENSOR_COLUMNS, reading)}
            for reading in raw_readings
        ]
        with image_path.open("rb") as handle:
            upload = assert_status(
                client.post(
                    "/upload", data={"image": (handle, image_path.name)},
                    content_type="multipart/form-data",
                ),
                201,
                f"{expected_status}-upload",
            )
        prediction = assert_status(
            client.post(
                "/predict",
                json={
                    "upload_id": upload["upload_id"],
                    "plant_type": CROP_LABELS[int(crops[index])],
                    "recent_sensor_readings": readings,
                },
            ),
            201,
            f"{expected_status}-predict",
        )
        assert prediction["analysis_status"] == expected_status
        assert 0 <= prediction["confidence"] <= 1
        assert prediction["cnn_feature_summary"]["top_regions"]
        assert prediction["observations"] and prediction["recommendations"]
        predictions[expected_status] = prediction

    assert any("too close" in item for item in predictions["inconclusive"]["observations"])

    non_leaf_path = PROJECT_ROOT / "frontend" / "src" / "assets" / "agrisense-ui-concept.png"
    with non_leaf_path.open("rb") as handle:
        rejected = assert_status(
            client.post(
                "/upload", data={"image": (handle, non_leaf_path.name)},
                content_type="multipart/form-data",
            ),
            422,
            "non-leaf-upload",
        )
    assert rejected["status"] in {"rejected", "retry_required"}
    assert rejected["reason_code"] != "valid_leaf"
    assert rejected["guidance"]

    history = assert_status(client.get("/history?page=1&limit=5"), 200, "history")
    assert history["total"] >= 1

    detail = assert_status(client.get(f"/history/{predictions['completed']['prediction_id']}"), 200, "history-detail")
    assert detail["prediction_id"] == predictions["completed"]["prediction_id"]

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
        "completed_prediction_id": predictions["completed"]["prediction_id"],
        "completed_class": predictions["completed"]["predicted_class"],
        "inconclusive_prediction_id": predictions["inconclusive"]["prediction_id"],
        "non_leaf_reason": rejected["reason_code"],
        "history_total": history["total"],
    }, indent=2))


if __name__ == "__main__":
    main()
