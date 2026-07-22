from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.analysis_service import AnalysisService  # noqa: E402


class AnalysisServiceTests(unittest.TestCase):
    def test_ambiguous_prediction_is_inconclusive(self) -> None:
        prediction = {
            "predicted_class": "Healthy",
            "probability_array": [0.27, 0.26, 0.25, 0.22],
            "crop_probabilities": {"Tomato": 0.7, "Apple": 0.3},
        }
        contract = {"decision_policy": {
            "min_confidence": 0.55, "min_margin": 0.12, "max_entropy": 1.2,
        }}

        analysis = AnalysisService.build(
            prediction, {"status": "accepted", "inferred_crop": "Tomato"},
            "Tomato", [], contract,
        )

        self.assertEqual(analysis["analysis_status"], "inconclusive")
        self.assertEqual(analysis["reliability"]["level"], "insufficient")
        self.assertTrue(any("Retake" in item for item in analysis["recommendations"]))

    def test_completed_analysis_reports_crop_mismatch(self) -> None:
        prediction = {
            "predicted_class": "High",
            "probability_array": [0.02, 0.03, 0.05, 0.90],
            "crop_probabilities": {"Apple": 0.8, "Tomato": 0.2},
        }
        contract = {"decision_policy": {
            "min_confidence": 0.55, "min_margin": 0.12, "max_entropy": 1.2,
        }}

        analysis = AnalysisService.build(
            prediction, {"status": "accepted", "inferred_crop": "Apple"},
            "Tomato", [], contract,
        )

        self.assertEqual(analysis["analysis_status"], "completed")
        self.assertFalse(analysis["crop_consistency"]["matches"])
        self.assertTrue(any("crop" in item.lower() for item in analysis["observations"]))


if __name__ == "__main__":
    unittest.main()
