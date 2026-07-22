"""Calibrated image-suitability and selective-prediction helpers."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import f1_score


def quality_metrics(image: np.ndarray) -> dict[str, float]:
    values = np.asarray(image, dtype="float32")
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("Quality metrics require an RGB image")
    if values.size and values.max() > 1.0:
        values = values / 255.0
    gray = values.mean(axis=-1)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(
            np.square(np.diff(gray, axis=1)).mean()
            + np.square(np.diff(gray, axis=0)).mean()
        ),
        "dark_clip": float((gray <= 0.02).mean()),
        "bright_clip": float((gray >= 0.98).mean()),
    }


def _normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, 1e-12)


def fit_leaf_reference(
    embeddings: np.ndarray,
    crop_indices: np.ndarray,
    quality_rows: list[dict[str, float]],
    negative_embeddings: np.ndarray,
    crop_labels: list[str],
    *,
    calibration_embeddings: np.ndarray | None = None,
    calibration_quality_rows: list[dict[str, float]] | None = None,
) -> dict:
    values = _normalize(embeddings)
    crops = np.asarray(crop_indices, dtype="int32")
    centroids = []
    for index, label in enumerate(crop_labels):
        selected = values[crops == index]
        if not len(selected):
            raise ValueError(f"No calibration embeddings for crop {label}")
        centroids.append(_normalize(selected.mean(axis=0, keepdims=True))[0])
    centroid_array = np.asarray(centroids)
    calibration_values = _normalize(
        embeddings if calibration_embeddings is None else calibration_embeddings
    )
    positive_scores = np.max(calibration_values @ centroid_array.T, axis=1)
    negative_values = _normalize(negative_embeddings)
    negative_scores = np.max(negative_values @ centroid_array.T, axis=1)
    negative_max = float(negative_scores.max())
    accept_threshold = float(min(1.0, max(np.quantile(positive_scores, 0.005), negative_max + 0.03)))
    retry_threshold = float(min(accept_threshold - 0.01, negative_max + 0.01))
    metric_names = ("brightness", "contrast", "sharpness", "dark_clip", "bright_clip")
    boundary_quality_rows = (
        quality_rows if calibration_quality_rows is None else calibration_quality_rows
    )
    quality = {
        name: np.asarray([row[name] for row in boundary_quality_rows])
        for name in metric_names
    }
    return {
        "version": 1,
        "crop_labels": list(crop_labels),
        "centroids": centroid_array.tolist(),
        "accept_threshold": accept_threshold,
        "retry_threshold": retry_threshold,
        "quality": {
            "min_brightness": float(np.quantile(quality["brightness"], 0.001)),
            "max_brightness": float(np.quantile(quality["brightness"], 0.999)),
            "min_contrast": float(np.quantile(quality["contrast"], 0.001)),
            "min_sharpness": float(np.quantile(quality["sharpness"], 0.001)),
            "max_dark_clip": float(np.quantile(quality["dark_clip"], 0.999)),
            "max_bright_clip": float(np.quantile(quality["bright_clip"], 0.999)),
        },
    }


def leaf_similarity(embedding: np.ndarray, reference: dict) -> tuple[float, str]:
    vector = _normalize(np.asarray(embedding).reshape(1, -1))[0]
    centroids = _normalize(np.asarray(reference["centroids"], dtype="float64"))
    scores = centroids @ vector
    index = int(np.argmax(scores))
    return float(scores[index]), str(reference["crop_labels"][index])


def _reliability_values(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(probabilities, dtype="float64")
    ordered = np.sort(values, axis=1)
    confidence = ordered[:, -1]
    margin = ordered[:, -1] - ordered[:, -2]
    entropy = -(values * np.log(np.clip(values, 1e-8, 1.0))).sum(axis=1)
    return confidence, margin, entropy


def fit_decision_policy(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    min_coverage: float = 0.70,
) -> dict[str, float]:
    confidence, margin, entropy = _reliability_values(probabilities)
    predicted = np.argmax(probabilities, axis=1)
    best = None
    for min_confidence in np.quantile(confidence, [0.0, 0.1, 0.2, 0.3]):
        for min_margin in np.quantile(margin, [0.0, 0.1, 0.2, 0.3]):
            for max_entropy in np.quantile(entropy, [0.7, 0.8, 0.9, 1.0]):
                mask = (confidence >= min_confidence) & (margin >= min_margin) & (entropy <= max_entropy)
                coverage = float(mask.mean())
                if coverage < min_coverage or not mask.any():
                    continue
                score = float(f1_score(y_true[mask], predicted[mask], average="macro", zero_division=0))
                candidate = (score, coverage, float(min_confidence), float(min_margin), float(max_entropy))
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
    if best is None:
        raise ValueError("No reliability policy satisfies minimum coverage")
    return {
        "min_confidence": best[2], "min_margin": best[3], "max_entropy": best[4],
        "validation_macro_f1": best[0], "validation_coverage": best[1],
    }


def classify_reliability(probabilities: np.ndarray, policy: dict) -> dict:
    values = np.asarray(probabilities, dtype="float64")
    if values.shape != (4,) or not np.isfinite(values).all() or not math.isclose(float(values.sum()), 1.0, abs_tol=1e-5):
        raise ValueError("Reliability requires one valid four-class probability row")
    confidence, margin, entropy = (item[0] for item in _reliability_values(values[None, :]))
    completed = (
        confidence >= float(policy["min_confidence"])
        and margin >= float(policy["min_margin"])
        and entropy <= float(policy["max_entropy"])
    )
    return {
        "analysis_status": "completed" if completed else "inconclusive",
        "level": "high" if completed and confidence >= 0.8 else "moderate" if completed else "insufficient",
        "confidence": float(confidence), "margin": float(margin), "entropy": float(entropy),
    }
