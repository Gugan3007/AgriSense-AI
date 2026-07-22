"""SQLAlchemy models for persisted uploads, predictions, and contact messages."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


class UploadedImage(db.Model):
    __tablename__ = "uploaded_images"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid4()))
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    image_path = db.Column(db.String(500), nullable=False)
    image_url = db.Column(db.String(500), nullable=False)
    content_type = db.Column(db.String(120), nullable=False)
    sensor_csv_path = db.Column(db.String(500), nullable=True)
    validation_result = db.Column(db.JSON, nullable=True)
    received_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)

    predictions = db.relationship("Prediction", back_populates="upload", lazy=True)

    def to_dict(self) -> dict:
        return {
            "upload_id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "image_url": self.image_url,
            "received_at": isoformat_utc(self.received_at),
            "has_sensor_csv": bool(self.sensor_csv_path),
            "image_validation": self.validation_result,
        }


class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid4()))
    upload_id = db.Column(db.String(36), db.ForeignKey("uploaded_images.id"), nullable=False)
    plant_type = db.Column(db.String(120), nullable=True)
    predicted_class = db.Column(db.String(40), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    class_probabilities = db.Column(db.JSON, nullable=False)
    cnn_feature_summary = db.Column(db.JSON, nullable=False)
    lstm_trend = db.Column(db.JSON, nullable=False)
    sensor_sequence = db.Column(db.JSON, nullable=False)
    sequence_adjustment = db.Column(db.JSON, nullable=False)
    prediction_time_ms = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)
    analysis_status = db.Column(db.String(32), nullable=True, default="completed")
    reliability = db.Column(db.JSON, nullable=True)
    image_validation = db.Column(db.JSON, nullable=True)
    crop_consistency = db.Column(db.JSON, nullable=True)
    observations = db.Column(db.JSON, nullable=True)
    recommendations = db.Column(db.JSON, nullable=True)

    upload = db.relationship("UploadedImage", back_populates="predictions")

    def to_dict(self) -> dict:
        return {
            "prediction_id": self.id,
            "upload_id": self.upload_id,
            "plant_type": self.plant_type,
            "image_url": self.upload.image_url if self.upload else None,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "class_probabilities": self.class_probabilities,
            "cnn_feature_summary": self.cnn_feature_summary,
            "lstm_trend": self.lstm_trend,
            "sensor_sequence": self.sensor_sequence,
            "sequence_adjustment": self.sequence_adjustment,
            "prediction_time_ms": self.prediction_time_ms,
            "timestamp": isoformat_utc(self.timestamp),
            "analysis_status": self.analysis_status or "completed",
            "reliability": self.reliability or {},
            "image_validation": self.image_validation or {},
            "crop_consistency": self.crop_consistency or {},
            "observations": self.observations or [],
            "recommendations": self.recommendations or [],
        }


class ContactMessage(db.Model):
    __tablename__ = "contact_messages"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid4()))
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)

    def to_dict(self) -> dict:
        return {
            "message_id": self.id,
            "name": self.name,
            "email": self.email,
            "created_at": isoformat_utc(self.created_at),
        }
