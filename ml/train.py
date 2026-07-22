"""Train the AgriSense image-first hybrid model and persist its artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from artifacts import build_contract, load_contract, promote_bundle
from calibration import fit_decision_policy, fit_leaf_reference, quality_metrics
from model import build_models, compile_model, configure_fine_tuning, save_model_summary
from preprocessing import add_auxiliary_targets, prepare_datasets
from utils import (
    CLASS_LABELS, MODEL_CONTRACT_PATH, MODEL_PATH, REPORTS_DIR, STAGED_CONTRACT_PATH,
    STAGED_MODEL_PATH, TENSORBOARD_DIR, ensure_directories, require_tensorflow,
    set_reproducible_seed, utc_now, write_json,
)


def _fit_calibration(tf, inference_model, data):
    embedding_model = tf.keras.Model(
        inference_model.inputs,
        inference_model.get_layer("cnn_global_pool").output,
    )
    embeddings, crops, quality_rows = [], [], []
    for inputs, targets in data.train:
        images = np.asarray(inputs["image"])
        embeddings.append(np.asarray(embedding_model.predict_on_batch(inputs)))
        crops.append(np.argmax(np.asarray(targets["crop"]), axis=1))
        quality_rows.extend(quality_metrics(image) for image in images)

    rng = np.random.default_rng(42)
    negative_images = [
        np.zeros((128, 128, 3), dtype="float32"),
        np.ones((128, 128, 3), dtype="float32"),
        rng.random((128, 128, 3), dtype="float32"),
        np.indices((128, 128)).sum(axis=0)[..., None].repeat(3, axis=2) % 2,
    ]
    concept_path = REPORTS_DIR.parents[1] / "frontend" / "src" / "assets" / "agrisense-ui-concept.png"
    with Image.open(concept_path) as image:
        negative_images.append(
            np.asarray(image.convert("RGB").resize((128, 128)), dtype="float32") / 255.0
        )
    negative_batch = np.asarray(negative_images, dtype="float32")
    sensor_batch = np.zeros((len(negative_batch), data.statistics["sequence_length"], 4), dtype="float32")
    negative_embeddings = embedding_model.predict(
        {"image": negative_batch, "sensor_sequence": sensor_batch}, verbose=0
    )
    true_rows, probability_rows = [], []
    calibration_embeddings, calibration_quality_rows = [], []
    for inputs, targets in data.validation:
        raw = inference_model.predict_on_batch(inputs)
        probability_rows.append(np.asarray(raw["stress_probabilities"]))
        true_rows.append(np.argmax(np.asarray(targets["stress"]), axis=1))
        images = np.asarray(inputs["image"])
        calibration_embeddings.append(
            np.asarray(embedding_model.predict_on_batch(inputs))
        )
        calibration_quality_rows.extend(quality_metrics(image) for image in images)
    reference = fit_leaf_reference(
        np.concatenate(embeddings), np.concatenate(crops), quality_rows,
        negative_embeddings, data.crop_labels,
        calibration_embeddings=np.concatenate(calibration_embeddings),
        calibration_quality_rows=calibration_quality_rows,
    )
    policy = fit_decision_policy(
        np.concatenate(true_rows), np.concatenate(probability_rows), min_coverage=0.70
    )
    return reference, policy


def plot_history(history: dict[str, list[float]]) -> None:
    for metric, filename, title in (
        ("stress_probabilities_accuracy", "accuracy_curve.png", "Model accuracy"),
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


def _balanced_class_weights(distribution: dict[str, int]) -> list[float]:
    total = sum(distribution.values())
    return [total / (len(CLASS_LABELS) * distribution[label]) for label in CLASS_LABELS]


def train(
    epochs: int = 18,
    fine_tune_epochs: int = 8,
    batch_size: int = 16,
    sequence_length: int = 7,
    max_steps=None,
    qa_only: bool = False,
):
    tf = require_tensorflow()
    ensure_directories()
    set_reproducible_seed()
    devices = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}; compute device: {'GPU ' + str(devices) if devices else 'CPU'}")
    data = prepare_datasets(sequence_length=sequence_length, batch_size=batch_size)
    training_model, inference_model = build_models(
        sequence_length=sequence_length,
        image_weights=None if qa_only else "imagenet",
    )
    compile_model(training_model, learning_rate=0.001)
    save_model_summary(inference_model)
    class_weights = _balanced_class_weights(data.statistics["label_distribution"]["train"])
    train_dataset = add_auxiliary_targets(data.train, class_weights)
    validation_dataset = add_auxiliary_targets(data.validation)
    warmup_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_stress_probabilities_loss", mode="min", patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_stress_probabilities_loss", mode="min", factor=0.5,
            patience=2, min_lr=1e-6,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(TENSORBOARD_DIR), histogram_freq=0),
    ]
    started = time.perf_counter()
    warmup = training_model.fit(
        train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=warmup_callbacks,
        verbose=1, steps_per_epoch=max_steps, validation_steps=max_steps, shuffle=False,
    )
    opened_layers = configure_fine_tuning(training_model, unfreeze_last=20)
    compile_model(training_model, learning_rate=1e-5)
    fine_tune_callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_stress_probabilities_macro_f1", mode="max", patience=4,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_stress_probabilities_macro_f1", mode="max", factor=0.5,
            patience=2, min_lr=1e-7,
        ),
    ]
    fine_tune = training_model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=fine_tune_epochs,
        callbacks=fine_tune_callbacks,
        verbose=1,
        steps_per_epoch=max_steps,
        validation_steps=max_steps,
        shuffle=False,
    )
    elapsed = time.perf_counter() - started
    history_values = {
        key: list(warmup.history.get(key, [])) + list(fine_tune.history.get(key, []))
        for key in set(warmup.history) | set(fine_tune.history)
    }
    frame = pd.DataFrame(history_values)
    frame.index.name = "epoch"
    frame.to_csv(REPORTS_DIR / "training_history.csv")
    plot_history(history_values)
    if qa_only:
        contract = build_contract(data.statistics)
    else:
        leaf_reference, decision_policy = _fit_calibration(tf, inference_model, data)
        contract = build_contract(data.statistics, leaf_reference, decision_policy)
    STAGED_MODEL_PATH.unlink(missing_ok=True)
    STAGED_CONTRACT_PATH.unlink(missing_ok=True)
    inference_model.save(STAGED_MODEL_PATH)
    write_json(STAGED_CONTRACT_PATH, contract)
    tf.keras.models.load_model(STAGED_MODEL_PATH, compile=False)
    load_contract(STAGED_CONTRACT_PATH)

    if qa_only:
        STAGED_MODEL_PATH.unlink(missing_ok=True)
        STAGED_CONTRACT_PATH.unlink(missing_ok=True)
        return {
            "qa_only": True,
            "completed_epochs": len(frame),
            "class_weights": class_weights,
            "fine_tuned_layers": opened_layers,
        }

    from evaluate import evaluate_model

    staged_evaluation = evaluate_model(
        sequence_length=sequence_length,
        batch_size=batch_size,
        model_path=STAGED_MODEL_PATH,
        contract_path=STAGED_CONTRACT_PATH,
        write_reports=False,
        enforce_model_acceptance=True,
    )
    promote_bundle(STAGED_MODEL_PATH, STAGED_CONTRACT_PATH, MODEL_PATH, MODEL_CONTRACT_PATH)
    metadata = {
        "model_name": inference_model.name,
        "architecture": "Fine-tuned MobileNetV2 image encoder with crop evidence + sensor LSTM + calibrated image-first probability fusion",
        "class_labels": CLASS_LABELS,
        "input_shapes": {"image": [128, 128, 3], "sensor_sequence": [sequence_length, 4]},
        "hyperparameters": {"learning_rate": 0.001, "fine_tune_learning_rate": 1e-5, "batch_size": batch_size, "requested_epochs": epochs, "requested_fine_tune_epochs": fine_tune_epochs, "completed_epochs": len(frame), "sequence_length": sequence_length, "optimizer": "Adam", "loss": "multi_output_categorical_crossentropy", "class_weights": class_weights, "image_probability_weight": 0.8, "sensor_probability_weight": 0.2, "fine_tuned_layers": opened_layers},
        "training_date": utc_now(),
        "training_seconds": round(elapsed, 2),
        "compute_device": "GPU" if devices else "CPU",
        "platform": platform.platform(),
        "parameter_count": int(inference_model.count_params()),
        "best_validation_accuracy": float(frame["val_stress_probabilities_accuracy"].max()),
        "final_validation_accuracy": float(frame["val_stress_probabilities_accuracy"].iloc[-1]),
        "staged_evaluation": staged_evaluation,
        "preprocessing": data.statistics,
    }
    write_json(REPORTS_DIR / "model_metadata.json", metadata)
    evaluate_model(sequence_length=sequence_length, batch_size=batch_size)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=None, help="QA-only limit per epoch")
    parser.add_argument("--qa-only", action="store_true", help="Validate one staged run without promotion")
    args = parser.parse_args()
    print(json.dumps(train(args.epochs, args.fine_tune_epochs, args.batch_size, args.sequence_length, args.max_steps, args.qa_only), indent=2))
