"""Upload endpoint."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from services.storage_service import save_upload

upload_bp = Blueprint("upload", __name__)


@upload_bp.post("/upload")
def upload_image():
    try:
        image = request.files.get("image") or request.files.get("file")
        sensor_csv = request.files.get("sensor_csv")
        record = save_upload(image, sensor_csv=sensor_csv)
        return jsonify({
            "upload_id": record.id,
            "image_url": record.image_url,
            "received_at": record.received_at.isoformat(),
        }), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
