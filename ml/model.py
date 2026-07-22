"""Functional Keras implementation of the required hybrid CNN-LSTM."""

from __future__ import annotations

from pathlib import Path

from utils import CLASS_LABELS, DEFAULT_SEQUENCE_LENGTH, IMAGE_SIZE, REPORTS_DIR, ensure_directories, require_tensorflow


def macro_f1_metric():
    """Return a serializable streaming macro-F1 metric without importing TF at module load."""
    tf = require_tensorflow()

    @tf.keras.utils.register_keras_serializable(package="AgriSense")
    class MacroF1(tf.keras.metrics.Metric):
        def __init__(self, num_classes=len(CLASS_LABELS), name="macro_f1", **kwargs):
            super().__init__(name=name, **kwargs)
            self.num_classes = num_classes
            self.matrix = self.add_weight(
                name="confusion_matrix", shape=(num_classes, num_classes), initializer="zeros"
            )

        def update_state(self, y_true, y_pred, sample_weight=None):
            true = tf.argmax(y_true, axis=-1)
            predicted = tf.argmax(y_pred, axis=-1)
            matrix = tf.math.confusion_matrix(
                true, predicted, num_classes=self.num_classes, dtype=self.dtype,
                weights=sample_weight,
            )
            self.matrix.assign_add(matrix)

        def result(self):
            true_positive = tf.linalg.diag_part(self.matrix)
            precision = tf.math.divide_no_nan(true_positive, tf.reduce_sum(self.matrix, axis=0))
            recall = tf.math.divide_no_nan(true_positive, tf.reduce_sum(self.matrix, axis=1))
            return tf.reduce_mean(tf.math.divide_no_nan(2 * precision * recall, precision + recall))

        def reset_state(self):
            self.matrix.assign(tf.zeros_like(self.matrix))

        def get_config(self):
            return {**super().get_config(), "num_classes": self.num_classes}

    return MacroF1()


def compile_model(model, learning_rate: float = 1e-3):
    """Compile the three-output training model with image-first supervision."""
    tf = require_tensorflow()
    losses = {
        name: tf.keras.losses.CategoricalCrossentropy()
        for name in ("stress_probabilities", "image_probabilities", "sensor_probabilities")
    }
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=losses,
        loss_weights={
            "stress_probabilities": 1.0,
            "image_probabilities": 0.5,
            "sensor_probabilities": 0.1,
        },
        metrics={
            "stress_probabilities": [
                tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
                macro_f1_metric(),
            ],
            "image_probabilities": [
                tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            ],
            "sensor_probabilities": [
                tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            ],
        },
    )
    return model


def build_models(
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    lstm_units: int = 128,
    dropout: float = 0.3,
    image_weights: str | None = "imagenet",
):
    """Return shared-weight training and production inference models."""
    tf = require_tensorflow()
    layers = tf.keras.layers

    image_input = layers.Input(
        shape=(*IMAGE_SIZE, 3), name="image"
    )
    sensor_input = layers.Input(shape=(sequence_length, 4), name="sensor_sequence")

    # MobileNetV2 expects pixels in [-1, 1]. The base stays frozen so the
    # limited project dataset trains a compact, stable visual classification head.
    image_scaled = layers.Rescaling(2.0, offset=-1.0, name="imagenet_rescaling")(image_input)
    image_encoder = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=image_weights,
        input_shape=(*IMAGE_SIZE, 3),
        name="image_encoder",
    )
    image_encoder.trainable = False
    image_map = image_encoder(image_scaled, training=False)
    image_features = layers.GlobalAveragePooling2D(name="cnn_global_pool")(image_map)
    image_features = layers.Dense(128, activation="relu", name="image_features")(image_features)
    image_features = layers.Dropout(0.2, name="image_dropout")(image_features)
    image_probabilities = layers.Dense(
        len(CLASS_LABELS), activation="softmax", name="image_probabilities"
    )(image_features)

    noisy_sensors = layers.GaussianNoise(0.15, name="sensor_noise")(sensor_input)
    sensor_features = layers.LSTM(
        min(lstm_units, 64), name="sensor_lstm"
    )(noisy_sensors)
    sensor_features = layers.Dense(32, activation="relu", name="sensor_features")(sensor_features)
    # A single dropout mask per sample removes the whole sensor representation
    # during training often enough to prevent sensor-only shortcut learning.
    sensor_features = layers.Dropout(
        dropout, noise_shape=(None, 1), name="sensor_modality_dropout"
    )(sensor_features)
    sensor_probabilities = layers.Dense(
        len(CLASS_LABELS), activation="softmax", name="sensor_probabilities"
    )(sensor_features)

    weighted_image = layers.Rescaling(0.8, name="image_probability_weight")(image_probabilities)
    weighted_sensor = layers.Rescaling(0.2, name="sensor_probability_weight")(sensor_probabilities)
    stress_probabilities = layers.Add(name="stress_probabilities")(
        [weighted_image, weighted_sensor]
    )

    inputs = {"image": image_input, "sensor_sequence": sensor_input}
    training_model = tf.keras.Model(
        inputs,
        {
            "stress_probabilities": stress_probabilities,
            "image_probabilities": image_probabilities,
            "sensor_probabilities": sensor_probabilities,
        },
        name="agrisense_training_model",
    )
    inference_model = tf.keras.Model(
        inputs, stress_probabilities, name="agrisense_image_first_cnn_lstm"
    )
    return training_model, inference_model


def build_model(
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    lstm_units: int = 128,
    dropout: float = 0.3,
    image_weights: str | None = "imagenet",
):
    """Compatibility wrapper returning the production inference model."""
    return build_models(sequence_length, lstm_units, dropout, image_weights)[1]


def save_model_summary(model, path: Path = REPORTS_DIR / "model_summary.txt") -> None:
    ensure_directories()
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    summary = "\n".join(lines) + "\n"
    print(summary)
    path.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    save_model_summary(build_model())
