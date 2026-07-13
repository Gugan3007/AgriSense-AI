"""Prediction history endpoints."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from models.db_models import Prediction

history_bp = Blueprint("history", __name__)


@history_bp.get("/history")
def history():
    try:
        page = max(int(request.args.get("page", 1)), 1)
        limit = min(max(int(request.args.get("limit", 10)), 1), 50)
    except ValueError:
        return jsonify({"error": "page and limit must be integers."}), 400

    pagination = Prediction.query.order_by(Prediction.timestamp.desc()).paginate(
        page=page, per_page=limit, error_out=False
    )
    return jsonify({
        "items": [item.to_dict() for item in pagination.items],
        "page": page,
        "limit": limit,
        "total": pagination.total,
        "pages": pagination.pages,
        "has_next": pagination.has_next,
        "has_prev": pagination.has_prev,
    })


@history_bp.get("/history/<prediction_id>")
def prediction_detail(prediction_id: str):
    prediction = Prediction.query.get(prediction_id)
    if prediction is None:
        return jsonify({"error": "Prediction not found."}), 404
    return jsonify(prediction.to_dict())
