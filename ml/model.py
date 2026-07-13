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
    tf = require_tensorflow()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            macro_f1_metric(),
        ],
    )
    return model


def build_model(
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    lstm_units: int = 128,
    dropout: float = 0.3,
):
    tf = require_tensorflow()
    layers = tf.keras.layers

    image_input = layers.Input(
        shape=(sequence_length, *IMAGE_SIZE, 3), name="image_sequence"
    )
    sensor_input = layers.Input(shape=(sequence_length, 4), name="sensor_sequence")

    # TimeDistributed applies the same CNN to each day, yielding one feature vector per timestep.
    x = layers.TimeDistributed(layers.Conv2D(32, 3, activation="relu"), name="td_conv_32")(image_input)
    x = layers.TimeDistributed(layers.MaxPooling2D(2), name="td_pool_1")(x)
    x = layers.TimeDistributed(layers.Conv2D(64, 3, activation="relu"), name="td_conv_64")(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(2), name="td_pool_2")(x)
    x = layers.TimeDistributed(layers.Conv2D(128, 3, activation="relu"), name="td_conv_128")(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(2), name="td_pool_3")(x)
    cnn_features = layers.TimeDistributed(layers.Flatten(), name="cnn_feature_vector")(x)

    # Sensors are aligned day-by-day with the corresponding visual feature vectors.
    fused = layers.Concatenate(axis=-1, name="image_sensor_fusion")([cnn_features, sensor_input])
    # The first LSTM preserves the temporal sequence; the second compresses the trend.
    temporal = layers.LSTM(lstm_units, return_sequences=True, name="lstm_sequence")(fused)
    temporal = layers.Dropout(dropout, name="lstm_sequence_dropout")(temporal)
    temporal = layers.LSTM(64, name="lstm_summary")(temporal)
    temporal = layers.Dropout(dropout, name="lstm_summary_dropout")(temporal)
    # Dense layers translate the learned temporal representation into four stress probabilities.
    head = layers.Dense(64, activation="relu", name="dense_64")(temporal)
    head = layers.Dropout(0.2, name="dense_dropout")(head)
    output = layers.Dense(len(CLASS_LABELS), activation="softmax", name="stress_probabilities")(head)
    return tf.keras.Model([image_input, sensor_input], output, name="agrisense_cnn_lstm")


def save_model_summary(model, path: Path = REPORTS_DIR / "model_summary.txt") -> None:
    ensure_directories()
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    summary = "\n".join(lines) + "\n"
    print(summary)
    path.write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    save_model_summary(build_model())
