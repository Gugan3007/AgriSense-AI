"""Leakage-safe sequence preprocessing for the hybrid CNN-LSTM."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import (
    CLASS_LABELS, DATASET_CSV, DEFAULT_SEQUENCE_LENGTH, IMAGE_SIZE, PROJECT_ROOT,
    REPORTS_DIR, SEED, SENSOR_COLUMNS, ensure_directories, require_tensorflow,
    set_reproducible_seed, write_json,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class PreparedData:
    train: object
    validation: object
    test: object
    statistics: dict
    split_plant_ids: dict[str, list[str]]
    sensor_normalization: dict[str, list[float]]


def load_valid_rows(csv_path: Path = DATASET_CSV) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Sequential dataset not found: {csv_path}")
    frame = pd.read_csv(csv_path, parse_dates=["Timestamp"])
    required = {"Plant_ID", "Image_Path", "Timestamp", "Stress_Level", *SENSOR_COLUMNS}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")
    invalid_labels = sorted(set(frame["Stress_Level"]) - set(CLASS_LABELS))
    if invalid_labels:
        raise ValueError(f"Unknown stress labels: {invalid_labels}")

    def resolve_image(value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else PROJECT_ROOT / path)

    frame["Resolved_Image_Path"] = frame["Image_Path"].map(resolve_image)
    exists = frame["Resolved_Image_Path"].map(lambda value: Path(value).is_file())
    for value in frame.loc[~exists, "Resolved_Image_Path"]:
        LOGGER.warning("Skipping missing image: %s", value)
    frame = frame.loc[exists].copy()
    if frame.empty:
        raise ValueError("No valid image rows remain after path verification")
    return frame.sort_values(["Plant_ID", "Timestamp"]).reset_index(drop=True)


def split_plant_ids(frame: pd.DataFrame, seed: int = SEED) -> dict[str, list[str]]:
    plant_ids = np.array(sorted(frame["Plant_ID"].unique()))
    if len(plant_ids) < 7:
        raise ValueError("At least seven plants are required for a 70/15/15 group split")
    train_ids, remainder = train_test_split(plant_ids, test_size=0.30, random_state=seed)
    validation_ids, test_ids = train_test_split(remainder, test_size=0.50, random_state=seed)
    result = {
        "train": sorted(train_ids.tolist()),
        "validation": sorted(validation_ids.tolist()),
        "test": sorted(test_ids.tolist()),
    }
    validate_split_integrity(frame, result)
    return result


def validate_split_integrity(
    frame: pd.DataFrame,
    splits: dict[str, list[str]],
    require_all_classes: bool = True,
) -> None:
    """Reject plant or source-image leakage and incomplete class partitions."""
    names = ("train", "validation", "test")
    plant_sets = {name: set(splits[name]) for name in names}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            overlap = plant_sets[left] & plant_sets[right]
            if overlap:
                raise ValueError(f"Plant leakage between {left} and {right}: {sorted(overlap)}")

    image_sets = {
        name: set(frame.loc[frame["Plant_ID"].isin(splits[name]), "Resolved_Image_Path"])
        for name in names
    }
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            overlap = image_sets[left] & image_sets[right]
            if overlap:
                sample = sorted(overlap)[:3]
                raise ValueError(
                    f"Source image leakage between {left} and {right}: {sample}"
                )

    if require_all_classes:
        required = set(CLASS_LABELS)
        for name in names:
            present = set(
                frame.loc[frame["Plant_ID"].isin(splits[name]), "Stress_Level"].astype(str)
            )
            missing = required - present
            if missing:
                raise ValueError(f"{name} split is missing stress classes: {sorted(missing)}")


def fit_sensor_normalization(frame: pd.DataFrame, train_ids: list[str]) -> dict[str, list[float]]:
    training = frame[frame["Plant_ID"].isin(train_ids)][SENSOR_COLUMNS].astype("float32")
    mean = training.mean().to_numpy(dtype="float32")
    std = training.std(ddof=0).replace(0, 1).to_numpy(dtype="float32")
    return {"columns": SENSOR_COLUMNS, "mean": mean.tolist(), "std": std.tolist()}


def make_windows(
    frame: pd.DataFrame,
    plant_ids: list[str],
    sequence_length: int,
    normalization: dict[str, list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths, sensors, labels = [], [], []
    mean = np.asarray(normalization["mean"], dtype="float32")
    std = np.asarray(normalization["std"], dtype="float32")
    label_to_index = {label: index for index, label in enumerate(CLASS_LABELS)}
    selected = frame[frame["Plant_ID"].isin(plant_ids)]
    for _, group in selected.groupby("Plant_ID", sort=True):
        group = group.sort_values("Timestamp")
        for start in range(len(group) - sequence_length + 1):
            window = group.iloc[start : start + sequence_length]
            paths.append(str(window.iloc[-1]["Resolved_Image_Path"]))
            raw_sensors = window[SENSOR_COLUMNS].to_numpy(dtype="float32")
            sensors.append((raw_sensors - mean) / std)
            labels.append(label_to_index[str(window.iloc[-1]["Stress_Level"])])
    if not paths:
        raise ValueError(f"No windows of length {sequence_length} could be created")
    return (
        np.asarray(paths, dtype=str),
        np.asarray(sensors, dtype="float32"),
        np.asarray(labels, dtype="int32"),
    )


def _decode_sample(path, sensor_sequence, label, augmenter=None):
    tf = require_tensorflow()

    def decode(path):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        image.set_shape((*IMAGE_SIZE, 3))
        return image

    image = decode(path)
    if augmenter is not None:
        # All randomness is confined to the training pipeline.
        image = augmenter(image, training=True)
        image = tf.clip_by_value(image, 0.0, 1.0)
    target = tf.one_hot(label, depth=len(CLASS_LABELS))
    return {"image": image, "sensor_sequence": sensor_sequence}, target


def build_tf_dataset(paths, sensors, labels, batch_size: int, training: bool):
    tf = require_tensorflow()
    augmenter = None
    if training:
        augmenter = tf.keras.Sequential(
            [
                tf.keras.layers.RandomRotation(0.08),
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomBrightness(0.12, value_range=(0.0, 1.0)),
                tf.keras.layers.RandomContrast(0.15),
            ],
            name="training_image_augmentation",
        )
    dataset = tf.data.Dataset.from_tensor_slices((paths, sensors, labels))
    if training:
        dataset = dataset.shuffle(len(labels), seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda p, s, y: _decode_sample(p, s, y, augmenter),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training,
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def prepare_datasets(
    csv_path: Path = DATASET_CSV,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    batch_size: int = 16,
    save_statistics: bool = True,
) -> PreparedData:
    ensure_directories()
    set_reproducible_seed()
    frame = load_valid_rows(csv_path)
    splits = split_plant_ids(frame)
    normalization = fit_sensor_normalization(frame, splits["train"])
    arrays = {
        name: make_windows(frame, ids, sequence_length, normalization)
        for name, ids in splits.items()
    }
    datasets = {
        name: build_tf_dataset(*values, batch_size=batch_size, training=name == "train")
        for name, values in arrays.items()
    }
    stats = {
        "csv_path": str(csv_path),
        "valid_rows": int(len(frame)),
        "missing_rows_skipped": int(len(pd.read_csv(csv_path)) - len(frame)),
        "image_size": [*IMAGE_SIZE, 3],
        "sequence_length": sequence_length,
        "sensor_columns": SENSOR_COLUMNS,
        "class_labels": CLASS_LABELS,
        "split_by": "Plant_ID",
        "plant_counts": {name: len(ids) for name, ids in splits.items()},
        "window_counts": {name: int(len(values[2])) for name, values in arrays.items()},
        "label_distribution": {
            name: {CLASS_LABELS[i]: int(np.sum(values[2] == i)) for i in range(len(CLASS_LABELS))}
            for name, values in arrays.items()
        },
        "augmentation": {
            "training_only": True,
            "operations": ["RandomRotation", "RandomFlip", "RandomBrightness", "RandomContrast"],
        },
        "plant_ids": splits,
        "sensor_normalization": normalization,
    }
    if save_statistics:
        write_json(REPORTS_DIR / "preprocessing_statistics.json", stats)
    return PreparedData(
        train=datasets["train"], validation=datasets["validation"], test=datasets["test"],
        statistics=stats, split_plant_ids=splits, sensor_normalization=normalization,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    prepared = prepare_datasets(sequence_length=args.sequence_length, batch_size=args.batch_size)
    print(json.dumps(prepared.statistics, indent=2))


if __name__ == "__main__":
    main()
