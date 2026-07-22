from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import create_app  # noqa: E402


class UploadRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = create_app().test_client()

    def test_unsuitable_decoded_image_returns_structured_422(self) -> None:
        stream = io.BytesIO()
        Image.new("RGB", (256, 256), "white").save(stream, format="PNG")
        stream.seek(0)
        blocked = {
            "status": "rejected", "reason_code": "non_leaf",
            "message": "This image does not look like a leaf.",
            "guidance": ["Upload one clear leaf."],
        }
        with patch(
            "services.storage_service.ImageValidationService.validate",
            return_value=blocked,
        ):
            response = self.client.post(
                "/upload", data={"image": (stream, "anything.png")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["reason_code"], "non_leaf")


if __name__ == "__main__":
    unittest.main()
