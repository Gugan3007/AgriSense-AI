from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from evaluate import (  # noqa: E402
    enforce_acceptance,
    evaluate_predictions,
    validate_probabilities,
)


class EvaluationTests(unittest.TestCase):
    def test_probability_validation_rejects_negative_values(self) -> None:
        probabilities = np.array([[1.1, -0.1, 0.0, 0.0]], dtype="float32")

        with self.assertRaisesRegex(ValueError, "outside"):
            validate_probabilities(probabilities)

    def test_probability_validation_rejects_rows_that_do_not_sum_to_one(self) -> None:
        probabilities = np.array([[0.2, 0.2, 0.2, 0.2]], dtype="float32")

        with self.assertRaisesRegex(ValueError, "sum"):
            validate_probabilities(probabilities)

    def test_evaluate_predictions_reports_multiclass_brier_score(self) -> None:
        y_true = np.array([0, 1, 2, 3], dtype="int32")
        probabilities = np.eye(4, dtype="float32")

        metrics = evaluate_predictions(y_true, probabilities)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1_macro"], 1.0)
        self.assertEqual(metrics["brier_score"], 0.0)

    def test_acceptance_requires_image_only_to_beat_sensor_only(self) -> None:
        metrics = {
            "normal": {"f1_macro": 0.7},
            "image_only": {"f1_macro": 0.4},
            "sensor_only": {"f1_macro": 0.8},
            "test_class_count": 4,
        }

        with self.assertRaisesRegex(RuntimeError, "image-only"):
            enforce_acceptance(metrics)


if __name__ == "__main__":
    unittest.main()
