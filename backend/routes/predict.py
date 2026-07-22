"""Prediction endpoint backed by the real trained image-first hybrid model."""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from models.db_models import Prediction, db
from services.inference_service import InferenceService
from services.analysis_service import AnalysisService
from services.image_validation_service import ImageValidationService
from services.storage_service import get_upload_or_none
from artifacts import load_contract

predict_bp = Blueprint("predict", __name__)
logger = logging.getLogger(__name__)


@predict_bp.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    upload_id = payload.get("upload_id")
    plant_type = payload.get("plant_type")

    if not upload_id:
        return jsonify({"error": "upload_id is required."}), 400

    upload = get_upload_or_none(upload_id)
    if upload is None:
        return jsonify({"error": f"No uploaded image found for upload_id={upload_id}."}), 404

    try:
        image_validation = ImageValidationService().validate(upload.image_path)
        if image_validation["status"] != "accepted":
            return jsonify({"error": image_validation["message"], **image_validation}), 422
        readings, adjustment = InferenceService.normalize_sensor_readings(
            payload.get("recent_sensor_readings", [])
        )
        prediction = InferenceService.predict(upload.image_path, readings)
        analysis = AnalysisService.build(
            prediction, image_validation, plant_type, readings, load_contract()
        )
        record = Prediction(
            upload_id=upload.id,
            plant_type=plant_type,
            predicted_class=prediction["predicted_class"],
            confidence=prediction["confidence"],
            class_probabilities=prediction["class_probabilities"],
            cnn_feature_summary=prediction["cnn_feature_summary"],
            lstm_trend=prediction["lstm_trend"],
            sensor_sequence=readings,
            sequence_adjustment=adjustment,
            prediction_time_ms=prediction["prediction_time_ms"],
            analysis_status=analysis["analysis_status"],
            reliability=analysis["reliability"],
            image_validation=analysis["image_validation"],
            crop_consistency=analysis["crop_consistency"],
            observations=analysis["observations"],
            recommendations=analysis["recommendations"],
        )
        db.session.add(record)
        db.session.commit()
        response = record.to_dict()
        response["sequence_adjustment"] = adjustment
        return jsonify(response), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("Prediction failed")
        db.session.rollback()
        return jsonify({"error": "Inference failed. Check backend logs for details."}), 500
