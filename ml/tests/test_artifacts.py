from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from artifacts import load_contract, promote_bundle, validate_contract  # noqa: E402
from utils import CLASS_LABELS, SENSOR_COLUMNS  # noqa: E402


class ArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.valid_contract = {
            "schema_version": 1,
            "class_labels": CLASS_LABELS,
            "sensor_columns": SENSOR_COLUMNS,
            "sensor_mean": [50.0, 25.0, 60.0, 15_000.0],
            "sensor_std": [10.0, 5.0, 10.0, 2_000.0],
            "image_size": [128, 128, 3],
            "sequence_length": 7,
            "input_names": ["image", "sensor_sequence"],
        }

    def test_contract_rejects_wrong_sensor_order(self) -> None:
        contract = {**self.valid_contract, "sensor_columns": list(reversed(SENSOR_COLUMNS))}

        with self.assertRaisesRegex(ValueError, "sensor column order"):
            validate_contract(contract)

    def test_contract_rejects_non_positive_standard_deviation(self) -> None:
        contract = {**self.valid_contract, "sensor_std": [10.0, 0.0, 10.0, 2_000.0]}

        with self.assertRaisesRegex(ValueError, "standard deviations"):
            validate_contract(contract)

    def test_promotion_replaces_model_and_contract_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged_model = root / "model.staging.keras"
            staged_contract = root / "preprocessing.staging.json"
            model_path = root / "model.keras"
            contract_path = root / "preprocessing.json"
            staged_model.write_bytes(b"new-model")
            staged_contract.write_text(json.dumps(self.valid_contract), encoding="utf-8")
            model_path.write_bytes(b"old-model")
            contract_path.write_text(
                json.dumps({**self.valid_contract, "schema_version": 0}), encoding="utf-8"
            )

            promote_bundle(staged_model, staged_contract, model_path, contract_path)

            self.assertEqual(model_path.read_bytes(), b"new-model")
            self.assertEqual(load_contract(contract_path)["schema_version"], 1)
            self.assertFalse(staged_model.exists())
            self.assertFalse(staged_contract.exists())

    def test_schema_v2_requires_calibrated_policies(self) -> None:
        contract = {
            **self.valid_contract,
            "schema_version": 2,
            "crop_labels": ["Apple", "Tomato"],
        }

        with self.assertRaisesRegex(ValueError, "leaf_validation"):
            validate_contract(contract)


if __name__ == "__main__":
    unittest.main()
