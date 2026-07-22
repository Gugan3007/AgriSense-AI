from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.image_validation_service import ImageValidationService  # noqa: E402


class ImageValidationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = {"leaf_validation": {
            "version": 1,
            "crop_labels": ["Tomato"],
            "centroids": [[1.0, 0.0]],
            "accept_threshold": 0.8,
            "retry_threshold": 0.2,
            "quality": {
                "min_brightness": 0.05, "max_brightness": 0.95,
                "min_contrast": 0.02, "min_sharpness": 0.0001,
                "max_dark_clip": 0.95, "max_bright_clip": 0.95,
            },
        }}
        rng = np.random.default_rng(42)
        self.textured = Image.fromarray(
            rng.integers(20, 230, size=(256, 256, 3), dtype=np.uint8)
        )

    def test_blank_image_is_rejected_before_embedding(self) -> None:
        service = ImageValidationService(
            contract=self.contract,
            embedding_provider=lambda _: self.fail("embedding should not run"),
        )

        result = service.validate(Image.new("RGB", (256, 256), "white"))

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason_code"], "near_uniform")

    def test_tiny_image_requires_retry(self) -> None:
        service = ImageValidationService(
            contract=self.contract, embedding_provider=lambda _: np.array([1.0, 0.0])
        )

        result = service.validate(self.textured.resize((32, 32)))

        self.assertEqual(result["status"], "retry_required")
        self.assertEqual(result["reason_code"], "too_small")

    def test_non_leaf_embedding_is_rejected(self) -> None:
        service = ImageValidationService(
            contract=self.contract, embedding_provider=lambda _: np.array([-1.0, 0.0])
        )

        result = service.validate(self.textured)

        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason_code"], "non_leaf")

    def test_clear_leaf_embedding_is_accepted(self) -> None:
        service = ImageValidationService(
            contract=self.contract, embedding_provider=lambda _: np.array([1.0, 0.0])
        )

        result = service.validate(self.textured)

        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["inferred_crop"], "Tomato")


if __name__ == "__main__":
    unittest.main()
