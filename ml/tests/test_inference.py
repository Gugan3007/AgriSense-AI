from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from inference import predict_stress  # noqa: E402


class _FakeModel:
    def predict(self, inputs, verbose=0):
        return {
            "stress_probabilities": np.array([[.05, .10, .80, .05]], dtype="float32"),
            "crop_probabilities": np.array([[.25, .75]], dtype="float32"),
        }


class InferenceTests(unittest.TestCase):
    @patch("inference.ModelSingleton.get", return_value=_FakeModel())
    @patch("inference.load_contract")
    def test_schema_v2_prediction_returns_named_crop_probabilities(self, contract, _model) -> None:
        contract.return_value = {
            "image_size": [128, 128, 3],
            "sequence_length": 7,
            "sensor_columns": ["a", "b", "c", "d"],
            "sensor_mean": [0., 0., 0., 0.],
            "sensor_std": [1., 1., 1., 1.],
            "crop_labels": ["Apple", "Tomato"],
        }

        result = predict_stress(
            np.ones((128, 128, 3), dtype="float32"),
            np.zeros((7, 4), dtype="float32"),
        )

        self.assertEqual(result["predicted_class"], "Medium")
        self.assertEqual(set(result["crop_probabilities"]), {"Apple", "Tomato"})
        self.assertAlmostEqual(result["crop_probabilities"]["Tomato"], .75)


if __name__ == "__main__":
    unittest.main()
