"""Contact form endpoint."""

from __future__ import annotations

import re

from flask import Blueprint, jsonify, request

from models.db_models import ContactMessage, db

contact_bp = Blueprint("contact", __name__)
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@contact_bp.post("/contact")
def contact():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip()
    email = str(payload.get("email", "")).strip()
    message = str(payload.get("message", "")).strip()

    if len(name) < 2:
        return jsonify({"error": "Name must be at least 2 characters."}), 400
    if not EMAIL_PATTERN.match(email):
        return jsonify({"error": "A valid email is required."}), 400
    if len(message) < 10:
        return jsonify({"error": "Message must be at least 10 characters."}), 400

    record = ContactMessage(name=name, email=email, message=message)
    db.session.add(record)
    db.session.commit()
    return jsonify({"success": True, **record.to_dict()}), 201
