"""Versioned preprocessing contracts and safe model-bundle promotion."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from utils import (
    CLASS_LABELS,
    DEFAULT_SEQUENCE_LENGTH,
    IMAGE_SIZE,
    MODEL_CONTRACT_PATH,
    MODEL_PATH,
    SENSOR_COLUMNS,
)

CONTRACT_SCHEMA_VERSION = 1
MODEL_INPUT_NAMES = ["image", "sensor_sequence"]


def build_contract(statistics: dict[str, Any]) -> dict[str, Any]:
    """Create the exact preprocessing contract paired with a trained model."""
    normalization = statistics["sensor_normalization"]
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "class_labels": list(CLASS_LABELS),
        "sensor_columns": list(normalization["columns"]),
        "sensor_mean": list(normalization["mean"]),
        "sensor_std": list(normalization["std"]),
        "image_size": [*IMAGE_SIZE, 3],
        "sequence_length": int(statistics["sequence_length"]),
        "input_names": list(MODEL_INPUT_NAMES),
    }
    validate_contract(contract)
    return contract


def validate_contract(contract: dict[str, Any]) -> None:
    """Validate ordering and dimensions that must match model inference."""
    required = {
        "schema_version",
        "class_labels",
        "sensor_columns",
        "sensor_mean",
        "sensor_std",
        "image_size",
        "sequence_length",
        "input_names",
    }
    missing = required - set(contract)
    if missing:
        raise ValueError(f"Preprocessing contract is missing fields: {sorted(missing)}")
    if contract["schema_version"] != CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported preprocessing schema version: {contract['schema_version']}"
        )
    if list(contract["class_labels"]) != CLASS_LABELS:
        raise ValueError("Preprocessing contract has an invalid class label order")
    if list(contract["sensor_columns"]) != SENSOR_COLUMNS:
        raise ValueError("Preprocessing contract has an invalid sensor column order")
    if list(contract["image_size"]) != [*IMAGE_SIZE, 3]:
        raise ValueError("Preprocessing contract has an invalid image size")
    if int(contract["sequence_length"]) != DEFAULT_SEQUENCE_LENGTH:
        raise ValueError("Preprocessing contract has an invalid sequence length")
    if list(contract["input_names"]) != MODEL_INPUT_NAMES:
        raise ValueError("Preprocessing contract has invalid model input names")

    means = [float(value) for value in contract["sensor_mean"]]
    deviations = [float(value) for value in contract["sensor_std"]]
    if len(means) != len(SENSOR_COLUMNS) or not all(math.isfinite(value) for value in means):
        raise ValueError("Preprocessing contract has invalid sensor means")
    if (
        len(deviations) != len(SENSOR_COLUMNS)
        or not all(math.isfinite(value) and value > 0 for value in deviations)
    ):
        raise ValueError("Preprocessing contract has invalid sensor standard deviations")


def load_contract(path: Path = MODEL_CONTRACT_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing preprocessing contract: {path}")
    contract = json.loads(path.read_text(encoding="utf-8"))
    validate_contract(contract)
    return contract


def promote_bundle(
    staged_model: Path,
    staged_contract: Path,
    model_path: Path = MODEL_PATH,
    contract_path: Path = MODEL_CONTRACT_PATH,
) -> None:
    """Promote a validated staged pair and roll back both targets on failure."""
    if not staged_model.is_file() or staged_model.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty staged model: {staged_model}")
    load_contract(staged_contract)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.parent.mkdir(parents=True, exist_ok=True)

    model_backup = model_path.with_suffix(model_path.suffix + ".backup")
    contract_backup = contract_path.with_suffix(contract_path.suffix + ".backup")
    for backup in (model_backup, contract_backup):
        backup.unlink(missing_ok=True)

    model_had_previous = model_path.exists()
    contract_had_previous = contract_path.exists()
    if model_had_previous:
        model_path.replace(model_backup)
    if contract_had_previous:
        contract_path.replace(contract_backup)

    try:
        staged_model.replace(model_path)
        staged_contract.replace(contract_path)
    except Exception:
        model_path.unlink(missing_ok=True)
        contract_path.unlink(missing_ok=True)
        if model_had_previous and model_backup.exists():
            model_backup.replace(model_path)
        if contract_had_previous and contract_backup.exists():
            contract_backup.replace(contract_path)
        raise
    else:
        model_backup.unlink(missing_ok=True)
        contract_backup.unlink(missing_ok=True)
