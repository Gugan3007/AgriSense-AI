"""Evaluate the saved model and produce all Phase 2 report artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize

from artifacts import load_contract
from calibration import classify_reliability, leaf_similarity, quality_metrics
from preprocessing import CROP_LABELS, prepare_datasets
from utils import (
    CLASS_LABELS, MODEL_CONTRACT_PATH, MODEL_PATH, REPORTS_DIR, ensure_directories,
    require_tensorflow, write_json,
)


def collect_predictions(model, dataset):
    true_batches, probability_batches = [], []
    for inputs, targets in dataset:
        raw = model.predict_on_batch(inputs)
        probabilities = raw["stress_probabilities"] if isinstance(raw, dict) else raw
        probability_batches.append(probabilities)
        stress_targets = targets["stress"] if isinstance(targets, dict) else targets
        true_batches.append(np.argmax(stress_targets.numpy(), axis=1))
    y_true = np.concatenate(true_batches)
    probabilities = np.concatenate(probability_batches)
    return y_true, probabilities, np.argmax(probabilities, axis=1)


def validate_probabilities(probabilities: np.ndarray) -> None:
    values = np.asarray(probabilities, dtype="float64")
    if values.ndim != 2 or values.shape[1] != len(CLASS_LABELS):
        raise ValueError(f"Expected probability shape (samples, {len(CLASS_LABELS)}), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Probabilities contain non-finite values")
    if np.any(values < -1e-7) or np.any(values > 1.0 + 1e-7):
        raise ValueError("Probabilities are outside the [0, 1] range")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Probability rows must sum to one")


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    validate_probabilities(probabilities)
    y_true = np.asarray(y_true, dtype="int32")
    y_pred = np.argmax(probabilities, axis=1)
    one_hot = np.eye(len(CLASS_LABELS), dtype="float32")[y_true]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "mean_confidence": float(np.mean(np.max(probabilities, axis=1))),
        "samples": int(len(y_true)),
    }


def enforce_acceptance(metrics: dict) -> None:
    failures = []
    if metrics.get("test_class_count") != len(CLASS_LABELS):
        failures.append("test data does not contain all four classes")
    normal = metrics["normal"]
    if float(normal["accuracy"]) <= 0.8192090395480226:
        failures.append("accuracy did not beat baseline")
    if float(normal["f1_macro"]) <= 0.7901860901925868:
        failures.append("macro-F1 did not beat baseline")
    if float(normal["brier_score"]) > 0.2740994393825531:
        failures.append("Brier score regressed")
    image_f1 = float(metrics["image_only"]["f1_macro"])
    sensor_f1 = float(metrics["sensor_only"]["f1_macro"])
    if image_f1 <= sensor_f1:
        failures.append(
            "image-only macro F1 must exceed sensor-only macro F1 "
            f"({image_f1:.4f} <= {sensor_f1:.4f})"
        )
    if float(metrics["leaf_validation"]["false_rejection_rate"]) > 0.05:
        failures.append("leaf false rejection exceeds 5%")
    if not metrics["negative_suite"]["all_blocked"]:
        failures.append("non-leaf suite was not fully blocked")
    if failures:
        raise RuntimeError("Model acceptance failed: " + "; ".join(failures))


def _counterfactual_probabilities(model, dataset, modality: str) -> np.ndarray:
    batches = []
    for inputs, _ in dataset:
        changed = {name: np.asarray(value) for name, value in inputs.items()}
        changed[modality] = np.roll(changed[modality], shift=1, axis=0)
        raw = model.predict_on_batch(changed)
        probabilities = raw["stress_probabilities"] if isinstance(raw, dict) else raw
        batches.append(np.asarray(probabilities, dtype="float32"))
    return np.concatenate(batches)


def _passes_leaf_gate(image: np.ndarray, embedding: np.ndarray, reference: dict) -> bool:
    metrics = quality_metrics(image)
    quality = reference["quality"]
    quality_ok = (
        metrics["brightness"] >= quality["min_brightness"]
        and metrics["brightness"] <= quality["max_brightness"]
        and metrics["contrast"] >= quality["min_contrast"]
        and metrics["sharpness"] >= quality["min_sharpness"]
        and metrics["dark_clip"] <= quality["max_dark_clip"]
        and metrics["bright_clip"] <= quality["max_bright_clip"]
    )
    similarity, _ = leaf_similarity(embedding, reference)
    return bool(quality_ok and similarity >= reference["accept_threshold"])


def _validation_metrics(tf, model, dataset, contract: dict) -> tuple[dict, dict, dict]:
    reference = contract["leaf_validation"]
    embedding_model = tf.keras.Model(model.inputs, model.get_layer("cnn_global_pool").output)
    leaf_results, crop_true, crop_pred = [], [], []
    for inputs, targets in dataset:
        embeddings = np.asarray(embedding_model.predict_on_batch(inputs))
        raw = model.predict_on_batch(inputs)
        crop_probabilities = np.asarray(raw["crop_probabilities"])
        images = np.asarray(inputs["image"])
        leaf_results.extend(
            _passes_leaf_gate(image, embedding, reference)
            for image, embedding in zip(images, embeddings)
        )
        crop_true.extend(np.argmax(np.asarray(targets["crop"]), axis=1))
        crop_pred.extend(np.argmax(crop_probabilities, axis=1))

    rng = np.random.default_rng(42)
    negatives = [
        np.zeros((128, 128, 3), dtype="float32"),
        np.ones((128, 128, 3), dtype="float32"),
        rng.random((128, 128, 3), dtype="float32"),
        (np.indices((128, 128)).sum(axis=0)[..., None].repeat(3, axis=2) % 2).astype("float32"),
    ]
    concept_path = REPORTS_DIR.parents[1] / "frontend" / "src" / "assets" / "agrisense-ui-concept.png"
    with Image.open(concept_path) as image:
        negatives.append(np.asarray(image.convert("RGB").resize((128, 128)), dtype="float32") / 255.0)
    negative_images = np.asarray(negatives)
    negative_sensors = np.zeros((len(negatives), contract["sequence_length"], 4), dtype="float32")
    negative_embeddings = embedding_model.predict(
        {"image": negative_images, "sensor_sequence": negative_sensors}, verbose=0
    )
    blocked = [
        not _passes_leaf_gate(image, embedding, reference)
        for image, embedding in zip(negative_images, negative_embeddings)
    ]
    return (
        {"false_rejection_rate": float(1.0 - np.mean(leaf_results)), "samples": len(leaf_results)},
        {"blocked": int(sum(blocked)), "samples": len(blocked), "all_blocked": bool(all(blocked))},
        {"accuracy": float(accuracy_score(crop_true, crop_pred)), "samples": len(crop_true), "class_count": len(set(crop_true))},
    )


def plot_confusion(y_true, y_pred) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_LABELS)))
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Greens")
    fig.colorbar(image, ax=axis)
    axis.set(xticks=range(4), yticks=range(4), xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
             xlabel="Predicted", ylabel="Actual", title="Test Confusion Matrix")
    for row in range(4):
        for column in range(4):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def plot_roc(y_true, probabilities) -> None:
    binary = label_binarize(y_true, classes=range(len(CLASS_LABELS)))
    fig, axis = plt.subplots(figsize=(7, 6))
    for index, label in enumerate(CLASS_LABELS):
        if len(np.unique(binary[:, index])) < 2:
            continue
        false_positive, true_positive, _ = roc_curve(binary[:, index], probabilities[:, index])
        axis.plot(false_positive, true_positive, label=f"{label} (AUC={auc(false_positive, true_positive):.3f})")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="One-vs-rest ROC curves")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "roc_curve.png", dpi=180)
    plt.close(fig)


def evaluate_model(
    sequence_length: int = 7,
    batch_size: int = 16,
    model_path=MODEL_PATH,
    contract_path=MODEL_CONTRACT_PATH,
    write_reports: bool = True,
    enforce_model_acceptance: bool = False,
) -> dict:
    tf = require_tensorflow()
    ensure_directories()
    model_path = Path(model_path)
    contract_path = Path(contract_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Train the model first; missing {model_path}")
    contract = load_contract(contract_path)
    data = prepare_datasets(sequence_length=sequence_length, batch_size=batch_size)
    normalization = data.sensor_normalization
    if not np.allclose(contract["sensor_mean"], normalization["mean"], atol=1e-6):
        raise ValueError("Saved sensor means do not match the evaluation dataset")
    if not np.allclose(contract["sensor_std"], normalization["std"], atol=1e-6):
        raise ValueError("Saved sensor standard deviations do not match the evaluation dataset")

    model = tf.keras.models.load_model(model_path, compile=False)
    y_true, probabilities, y_pred = collect_predictions(model, data.test)
    image_model = tf.keras.Model(model.inputs, model.get_layer("image_probabilities").output)
    sensor_model = tf.keras.Model(model.inputs, model.get_layer("sensor_probabilities").output)
    image_true, image_probabilities, _ = collect_predictions(image_model, data.test)
    sensor_true, sensor_probabilities, _ = collect_predictions(sensor_model, data.test)
    if not np.array_equal(y_true, image_true) or not np.array_equal(y_true, sensor_true):
        raise AssertionError("Evaluation datasets changed order between modality passes")

    shuffled_images = _counterfactual_probabilities(model, data.test, "image")
    shuffled_sensors = _counterfactual_probabilities(model, data.test, "sensor_sequence")
    normal = evaluate_predictions(y_true, probabilities)
    metrics = {
        **normal,
        "test_samples": int(len(y_true)),
        "test_class_count": int(len(np.unique(y_true))),
        "normal": normal,
        "image_only": evaluate_predictions(y_true, image_probabilities),
        "sensor_only": evaluate_predictions(y_true, sensor_probabilities),
        "counterfactual_sensitivity": {
            "image_mean_total_variation": float(
                np.mean(0.5 * np.sum(np.abs(probabilities - shuffled_images), axis=1))
            ),
            "sensor_mean_total_variation": float(
                np.mean(0.5 * np.sum(np.abs(probabilities - shuffled_sensors), axis=1))
            ),
        },
    }
    if contract.get("schema_version") == 2:
        leaf_metrics, negative_metrics, crop_metrics = _validation_metrics(
            tf, model, data.test, contract
        )
        decisions = [
            classify_reliability(row, contract["decision_policy"])
            for row in probabilities
        ]
        selected = np.asarray([item["analysis_status"] == "completed" for item in decisions])
        selective = {
            "coverage": float(selected.mean()),
            "samples": int(selected.sum()),
        }
        if selected.any():
            selective.update(evaluate_predictions(y_true[selected], probabilities[selected]))
        metrics.update({
            "leaf_validation": leaf_metrics,
            "negative_suite": negative_metrics,
            "crop": crop_metrics,
            "selective": selective,
        })
    if enforce_model_acceptance:
        enforce_acceptance(metrics)

    if not write_reports:
        return metrics

    report = classification_report(
        y_true, y_pred, labels=range(4), target_names=CLASS_LABELS, zero_division=0
    )
    (REPORTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    write_json(REPORTS_DIR / "evaluation_metrics.json", metrics)
    pd.DataFrame(probabilities, columns=[f"probability_{x}" for x in CLASS_LABELS]).assign(
        actual=[CLASS_LABELS[x] for x in y_true], predicted=[CLASS_LABELS[x] for x in y_pred]
    ).to_csv(REPORTS_DIR / "test_predictions.csv", index=False)
    plot_confusion(y_true, y_pred)
    plot_roc(y_true, probabilities)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    evaluate_model(args.sequence_length, args.batch_size)
