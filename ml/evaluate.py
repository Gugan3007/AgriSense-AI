"""Evaluate the saved model and produce all Phase 2 report artifacts."""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize

from model import macro_f1_metric
from preprocessing import prepare_datasets
from utils import CLASS_LABELS, MODEL_PATH, REPORTS_DIR, ensure_directories, require_tensorflow, write_json


def collect_predictions(model, dataset):
    true_batches, probability_batches = [], []
    for inputs, targets in dataset:
        probability_batches.append(model.predict_on_batch(inputs))
        true_batches.append(np.argmax(targets.numpy(), axis=1))
    y_true = np.concatenate(true_batches)
    probabilities = np.concatenate(probability_batches)
    return y_true, probabilities, np.argmax(probabilities, axis=1)


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


def evaluate_model(sequence_length: int = 7, batch_size: int = 16) -> dict:
    tf = require_tensorflow()
    ensure_directories()
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Train the model first; missing {MODEL_PATH}")
    data = prepare_datasets(sequence_length=sequence_length, batch_size=batch_size)
    custom_objects = {"MacroF1": type(macro_f1_metric())}
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    y_true, probabilities, y_pred = collect_predictions(model, data.test)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_samples": int(len(y_true)),
    }
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
