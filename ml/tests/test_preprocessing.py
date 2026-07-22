from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from preprocessing import make_windows, validate_split_integrity  # noqa: E402
from utils import CLASS_LABELS, DATASET_CSV, SENSOR_COLUMNS  # noqa: E402


class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for index in range(7):
            rows.append({
                "Plant_ID": "P1",
                "Timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(index, unit="D"),
                "Resolved_Image_Path": f"/images/leaf-{index}.jpg",
                "Stress_Level": CLASS_LABELS[index // 2 if index < 6 else 3],
                "Soil_Moisture": 60.0 - index,
                "Temperature": 24.0 + index,
                "Humidity": 65.0 - index,
                "Light_Intensity": 14_000.0 + index * 100,
            })
        self.frame = pd.DataFrame(rows)
        self.normalization = {
            "columns": SENSOR_COLUMNS,
            "mean": [50.0, 25.0, 60.0, 15_000.0],
            "std": [10.0, 5.0, 10.0, 2_000.0],
        }

    def test_make_windows_uses_only_the_current_leaf_image(self) -> None:
        images, sensors, labels = make_windows(
            self.frame, ["P1"], sequence_length=7, normalization=self.normalization
        )

        self.assertEqual(images.shape, (1,))
        self.assertEqual(images[0], "/images/leaf-6.jpg")
        self.assertEqual(sensors.shape, (1, 7, 4))
        self.assertEqual(labels.tolist(), [3])

    def test_split_integrity_rejects_source_image_overlap(self) -> None:
        frame = pd.DataFrame({
            "Plant_ID": ["P1", "P2", "P3"],
            "Resolved_Image_Path": ["/images/shared.jpg", "/images/shared.jpg", "/images/test.jpg"],
            "Stress_Level": ["Healthy", "Low", "High"],
        })
        splits = {"train": ["P1"], "validation": ["P2"], "test": ["P3"]}

        with self.assertRaisesRegex(ValueError, "image leakage"):
            validate_split_integrity(frame, splits, require_all_classes=False)

    def test_checked_in_dataset_has_unique_source_images(self) -> None:
        frame = pd.read_csv(DATASET_CSV)

        self.assertEqual(
            int(frame["Image_Path"].duplicated().sum()),
            0,
            "generated dataset must not reuse a source image",
        )


if __name__ == "__main__":
    unittest.main()
