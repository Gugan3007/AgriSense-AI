from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.inference_service import InferenceService  # noqa: E402


class InferenceServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.readings = [
            {
                "Soil_Moisture": 62.0 - index * 3.0,
                "Temperature": 24.0 + index,
                "Humidity": 67.0 - index * 2.0,
                "Light_Intensity": 14_500.0 + index * 500.0,
            }
            for index in range(7)
        ]

    def test_temporal_scores_are_bounded(self) -> None:
        result = InferenceService.lstm_trend(self.readings)

        self.assertEqual(len(result["stress_score"]), 7)
        self.assertTrue(all(0.0 <= value <= 100.0 for value in result["stress_score"]))
        self.assertNotIn("stress_proxy", result)
        self.assertEqual(result["direction"], "declining")

    def test_non_finite_sensor_value_is_rejected(self) -> None:
        readings = [{**self.readings[0], "Temperature": float("nan")}]

        with self.assertRaisesRegex(ValueError, "finite"):
            InferenceService.normalize_sensor_readings(readings)

    def test_sensor_value_outside_physical_range_is_rejected(self) -> None:
        readings = [{**self.readings[0], "Soil_Moisture": -1.0}]

        with self.assertRaisesRegex(ValueError, "between 5 and 100"):
            InferenceService.normalize_sensor_readings(readings)


if __name__ == "__main__":
    unittest.main()
