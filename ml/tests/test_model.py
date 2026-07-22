from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from model import build_models, configure_fine_tuning  # noqa: E402


class ModelArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.training_model, cls.inference_model = build_models(image_weights=None)
        cls.sample_inputs = {
            "image": np.zeros((1, 128, 128, 3), dtype="float32"),
            "sensor_sequence": np.zeros((1, 7, 4), dtype="float32"),
        }

    def test_inference_model_uses_single_image_and_sensor_sequence(self) -> None:
        self.assertEqual(
            tuple(self.inference_model.get_layer("image").batch_shape[1:]),
            (128, 128, 3),
        )
        self.assertEqual(
            tuple(self.inference_model.get_layer("sensor_sequence").batch_shape[1:]),
            (7, 4),
        )

    def test_training_model_exposes_auxiliary_modality_heads(self) -> None:
        self.assertEqual(
            set(self.training_model.output_names),
            {
                "stress_probabilities", "image_probabilities",
                "sensor_probabilities", "crop_probabilities",
            },
        )

    def test_inference_model_exposes_stress_and_crop_evidence(self) -> None:
        self.assertEqual(
            set(self.inference_model.output_names),
            {"stress_probabilities", "crop_probabilities"},
        )

    def test_fine_tuning_excludes_batch_normalization(self) -> None:
        opened = configure_fine_tuning(self.training_model, unfreeze_last=20)

        self.assertGreater(len(opened), 0)
        self.assertTrue(all("batch_normalization" not in name for name in opened))

    def test_fused_output_is_an_image_first_probability_mixture(self) -> None:
        outputs = self.training_model(self.sample_inputs, training=False)
        expected = 0.8 * outputs["image_probabilities"] + 0.2 * outputs["sensor_probabilities"]
        probabilities = self.inference_model(
            self.sample_inputs, training=False
        )["stress_probabilities"].numpy()

        np.testing.assert_allclose(probabilities, expected.numpy(), atol=1e-6)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
        self.assertTrue(np.all(probabilities >= 0.0))
        self.assertTrue(np.all(probabilities <= 1.0))

    def test_model_is_smaller_than_the_previous_flatten_architecture(self) -> None:
        self.assertLess(self.inference_model.count_params(), 4_000_000)

    def test_requested_lstm_units_are_not_silently_capped(self) -> None:
        _, model = build_models(lstm_units=128, image_weights=None)

        self.assertEqual(model.get_layer("sensor_lstm").units, 128)


if __name__ == "__main__":
    unittest.main()
