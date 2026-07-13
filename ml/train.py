"""Train the AgriSense CNN-LSTM and persist its real artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from evaluate import evaluate_model
from model import build_model, compile_model, save_model_summary
from preprocessing import prepare_datasets
from utils import (
    CLASS_LABELS, MODEL_PATH, REPORTS_DIR, TENSORBOARD_DIR, ensure_directories,
    require_tensorflow, set_reproducible_seed, utc_now, write_json,
)


def plot_history(history: dict[str, list[float]]) -> None:
    for metric, filename, title in (
        ("accuracy", "accuracy_curve.png", "Model accuracy"),
        ("loss", "loss_curve.png", "Model loss"),
    ):
        fig, axis = plt.subplots(figsize=(8, 5))
        axis.plot(history[metric], label="Training")
        axis.plot(history[f"val_{metric}"], label="Validation")
        axis.set(xlabel="Epoch", ylabel=metric.title(), title=title)
        axis.legend()
        axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / filename, dpi=180)
        plt.close(fig)


def train(epochs: int = 18, batch_size: int = 16, sequence_length: int = 7, max_steps=None):
    tf = require_tensorflow()
    ensure_directories()
    set_reproducible_seed()
    devices = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}; compute device: {'GPU ' + str(devices) if devices else 'CPU'}")
    data = prepare_datasets(sequence_length=sequence_length, batch_size=batch_size)
    model = compile_model(build_model(sequence_length=sequence_length), learning_rate=0.001)
    save_model_summary(model)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir=str(TENSORBOARD_DIR), histogram_freq=0),
    ]
    started = time.perf_counter()
    history = model.fit(
        data.train, validation_data=data.validation, epochs=epochs, callbacks=callbacks,
        verbose=1, steps_per_epoch=max_steps, validation_steps=max_steps,
    )
    elapsed = time.perf_counter() - started
    frame = pd.DataFrame(history.history)
    frame.index.name = "epoch"
    frame.to_csv(REPORTS_DIR / "training_history.csv")
    plot_history(history.history)
    metadata = {
        "model_name": model.name,
        "architecture": "TimeDistributed CNN + sensor fusion + stacked LSTM + softmax",
        "class_labels": CLASS_LABELS,
        "input_shapes": {"image_sequence": [sequence_length, 128, 128, 3], "sensor_sequence": [sequence_length, 4]},
        "hyperparameters": {"learning_rate": 0.001, "batch_size": batch_size, "requested_epochs": epochs, "completed_epochs": len(frame), "sequence_length": sequence_length, "optimizer": "Adam", "loss": "categorical_crossentropy"},
        "training_date": utc_now(),
        "training_seconds": round(elapsed, 2),
        "compute_device": "GPU" if devices else "CPU",
        "platform": platform.platform(),
        "parameter_count": int(model.count_params()),
        "best_validation_accuracy": float(frame["val_accuracy"].max()),
        "final_validation_accuracy": float(frame["val_accuracy"].iloc[-1]),
        "preprocessing": data.statistics,
    }
    write_json(REPORTS_DIR / "model_metadata.json", metadata)
    evaluate_model(sequence_length=sequence_length, batch_size=batch_size)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=None, help="QA-only limit per epoch")
    args = parser.parse_args()
    print(json.dumps(train(args.epochs, args.batch_size, args.sequence_length, args.max_steps), indent=2))
