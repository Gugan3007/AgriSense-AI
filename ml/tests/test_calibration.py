from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from calibration import (  # noqa: E402
    classify_reliability,
    fit_leaf_reference,
    leaf_similarity,
    quality_metrics,
)


class CalibrationTests(unittest.TestCase):
    def test_quality_metrics_identify_uniform_image(self) -> None:
        metrics = quality_metrics(np.ones((128, 128, 3), dtype="float32"))

        self.assertEqual(metrics["contrast"], 0.0)
        self.assertEqual(metrics["sharpness"], 0.0)
        self.assertEqual(metrics["bright_clip"], 1.0)

    def test_leaf_reference_separates_leaf_and_negative(self) -> None:
        qualities = [
            {"brightness": .5, "contrast": .2, "sharpness": .1, "dark_clip": 0., "bright_clip": 0.}
            for _ in range(4)
        ]
        reference = fit_leaf_reference(
            np.array([[1., 0.], [.98, .02], [0., 1.], [.02, .98]]),
            np.array([0, 0, 1, 1]), qualities,
            np.array([[-1., 0.], [0., -1.]]), ["Apple", "Tomato"],
        )

        self.assertGreaterEqual(
            leaf_similarity(np.array([1., 0.]), reference)[0],
            reference["accept_threshold"],
        )
        self.assertLess(
            leaf_similarity(np.array([-1., 0.]), reference)[0],
            reference["retry_threshold"],
        )

    def test_leaf_reference_uses_clean_calibration_rows_for_acceptance_bounds(self) -> None:
        training_qualities = [
            {"brightness": .5, "contrast": .2, "sharpness": .1, "dark_clip": 0., "bright_clip": 0.}
            for _ in range(4)
        ]
        calibration_embeddings = np.array(
            [[1., 0.], [.8, .6], [0., 1.], [.6, .8]], dtype="float32"
        )
        calibration_qualities = [
            {"brightness": .2, "contrast": .1, "sharpness": .05, "dark_clip": .1, "bright_clip": .02},
            {"brightness": .8, "contrast": .3, "sharpness": .20, "dark_clip": .0, "bright_clip": .00},
            {"brightness": .3, "contrast": .2, "sharpness": .10, "dark_clip": .0, "bright_clip": .01},
            {"brightness": .7, "contrast": .2, "sharpness": .15, "dark_clip": .0, "bright_clip": .00},
        ]

        reference = fit_leaf_reference(
            np.array([[1., 0.], [.98, .02], [0., 1.], [.02, .98]]),
            np.array([0, 0, 1, 1]),
            training_qualities,
            np.array([[-1., 0.], [0., -1.]]),
            ["Apple", "Tomato"],
            calibration_embeddings=calibration_embeddings,
            calibration_quality_rows=calibration_qualities,
        )

        centroids = np.asarray(reference["centroids"])
        calibration_scores = np.max(
            calibration_embeddings
            / np.linalg.norm(calibration_embeddings, axis=1, keepdims=True)
            @ centroids.T,
            axis=1,
        )
        self.assertAlmostEqual(
            reference["accept_threshold"],
            float(np.quantile(calibration_scores, 0.005)),
        )
        self.assertLess(reference["quality"]["min_brightness"], .3)

    def test_reliability_abstains_on_ambiguous_probabilities(self) -> None:
        result = classify_reliability(
            np.array([.27, .26, .25, .22]),
            {"min_confidence": .55, "min_margin": .12, "max_entropy": 1.20},
        )

        self.assertEqual(result["analysis_status"], "inconclusive")
        self.assertEqual(result["level"], "insufficient")


if __name__ == "__main__":
    unittest.main()
