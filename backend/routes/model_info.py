"""Model metadata endpoint."""

from __future__ import annotations

import json

from flask import Blueprint, jsonify

from config import MODEL_METADATA_PATH, ML_REPORTS_DIR, MODEL_PATH

model_info_bp = Blueprint("model_info", __name__)


@model_info_bp.get("/model-info")
def model_info():
    if not MODEL_METADATA_PATH.is_file():
        return jsonify({"error": "model_metadata.json not found. Train the model first."}), 404
    metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    report_files = {
        "accuracy_curve": "accuracy_curve.png",
        "loss_curve": "loss_curve.png",
        "confusion_matrix": "confusion_matrix.png",
        "roc_curve": "roc_curve.png",
        "classification_report": "classification_report.txt",
        "training_history": "training_history.csv",
        "tuning_results": "tuning_results.csv",
    }
    return jsonify({
        **metadata,
        "model_file": str(MODEL_PATH),
        "report_urls": {
            key: f"/reports/{filename}"
            for key, filename in report_files.items()
            if (ML_REPORTS_DIR / filename).is_file()
        },
    })
