"""File upload validation and persistence helpers."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from config import Config, UPLOAD_DIR
from models.db_models import UploadedImage, db
from services.image_validation_service import ImageValidationError, ImageValidationService


def _extension(filename: str) -> str:
    return Path(filename or "").suffix.lower().lstrip(".")


def is_allowed_image(filename: str) -> bool:
    return _extension(filename) in Config.ALLOWED_IMAGE_EXTENSIONS


def is_allowed_sensor_csv(filename: str) -> bool:
    return _extension(filename) in Config.ALLOWED_SENSOR_EXTENSIONS


def save_upload(image_file: FileStorage, sensor_csv: FileStorage | None = None) -> tuple[UploadedImage, dict]:
    if not image_file or not image_file.filename:
        raise ValueError("An image file is required.")
    if not is_allowed_image(image_file.filename):
        raise ValueError("Unsupported image type. Use JPG, PNG, or WebP.")

    upload_id = str(uuid4())
    original_filename = secure_filename(image_file.filename)
    filename = f"{upload_id}.jpg"
    image_path = UPLOAD_DIR / filename

    try:
        with Image.open(image_file.stream) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Unsupported decoded image format. Use JPG, PNG, or WebP.")
            if getattr(image, "is_animated", False):
                raise ValueError("Animated images are not supported.")
            converted = ImageOps.exif_transpose(image).convert("RGB")
            validation = ImageValidationService().validate(converted)
            if validation["status"] != "accepted":
                raise ImageValidationError(validation)
            converted.save(image_path, format="JPEG", quality=92)
    except ImageValidationError:
        raise
    except ValueError:
        raise
    except Exception as exc:  # Pillow raises many concrete image parsing exceptions.
        raise ValueError("Uploaded file is not a readable image.") from exc

    sensor_csv_path: str | None = None
    if sensor_csv and sensor_csv.filename:
        if not is_allowed_sensor_csv(sensor_csv.filename):
            raise ValueError("Unsupported sensor file type. Use CSV.")
        sensor_filename = f"{upload_id}_sensors.csv"
        destination = UPLOAD_DIR / sensor_filename
        sensor_csv.save(destination)
        sensor_csv_path = str(destination)

    record = UploadedImage(
        id=upload_id,
        filename=filename,
        original_filename=original_filename,
        image_path=str(image_path),
        image_url=f"/static/uploads/{filename}",
        content_type="image/jpeg",
        sensor_csv_path=sensor_csv_path,
        validation_result=validation,
    )
    db.session.add(record)
    db.session.commit()
    return record, validation


def get_upload_or_none(upload_id: str) -> UploadedImage | None:
    if not upload_id:
        return None
    return db.session.get(UploadedImage, upload_id)
