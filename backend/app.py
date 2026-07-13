"""Flask entrypoint for the AgriSense AI API."""

from __future__ import annotations

import logging

from pathlib import Path

from flask import Flask, jsonify, send_file, send_from_directory
from flask_cors import CORS

from config import Config, ML_REPORTS_DIR, PROJECT_ROOT, ensure_backend_directories
from models.db_models import db
from routes.contact import contact_bp
from routes.dataset_info import dataset_info_bp
from routes.history import history_bp
from routes.model_info import model_info_bp
from routes.predict import predict_bp
from routes.upload import upload_bp


def create_app() -> Flask:
    ensure_backend_directories()
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.config.from_object(Config)
    CORS(app, origins=Config.CORS_ORIGINS)
    db.init_app(app)

    app.register_blueprint(upload_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(model_info_bp)
    app.register_blueprint(dataset_info_bp)
    app.register_blueprint(contact_bp)

    with app.app_context():
        db.create_all()

    @app.get("/")
    def root():
        return jsonify({
            "name": "AgriSense AI API",
            "slogan": "Predict Today. Protect Tomorrow.",
            "status": "ok",
        })

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/reports/<path:filename>")
    def reports(filename: str):
        return send_from_directory(ML_REPORTS_DIR, filename)

    @app.get("/dataset-image/<path:image_path>")
    def dataset_image(image_path: str):
        candidate = (PROJECT_ROOT / image_path).resolve()
        dataset_root = (PROJECT_ROOT / "dataset").resolve()
        if dataset_root not in candidate.parents or not candidate.is_file():
            return jsonify({"error": "Dataset image not found."}), 404
        return send_file(candidate)

    @app.errorhandler(413)
    def file_too_large(_error):
        return jsonify({"error": "File too large. Maximum upload size is 8 MB."}), 413

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"error": "Endpoint not found."}), 404

    @app.errorhandler(500)
    def internal_error(_error):
        logging.exception("Unhandled backend error")
        return jsonify({"error": "Internal server error."}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
