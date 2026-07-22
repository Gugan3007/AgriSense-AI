"""Reproducible lightweight random search for the required hyperparameter space."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time

import pandas as pd

from model import build_models, compile_model
from preprocessing import add_auxiliary_targets, prepare_datasets
from utils import REPORTS_DIR, SEED, ensure_directories, require_tensorflow, set_reproducible_seed

SEARCH_SPACE = {
    "lstm_units": [64, 128],
    "dropout": [0.2, 0.3, 0.4],
    "learning_rate": [0.001, 0.0005],
    "sequence_length": [5, 7, 10],
}


def tune(trials: int = 6, epochs: int = 3, batch_size: int = 16) -> pd.DataFrame:
    tf = require_tensorflow()
    ensure_directories()
    set_reproducible_seed()
    configurations = [
        dict(zip(SEARCH_SPACE, values))
        for values in itertools.product(*(SEARCH_SPACE[key] for key in SEARCH_SPACE))
    ]
    selected = random.Random(SEED).sample(configurations, min(trials, len(configurations)))
    results = []
    for index, config in enumerate(selected, start=1):
        print(f"Trial {index}/{len(selected)}: {config}")
        tf.keras.backend.clear_session()
        data = prepare_datasets(
            sequence_length=config["sequence_length"], batch_size=batch_size, save_statistics=False
        )
        training_model, _ = build_models(
            config["sequence_length"], config["lstm_units"], config["dropout"]
        )
        model = compile_model(training_model, config["learning_rate"])
        started = time.perf_counter()
        history = model.fit(
            add_auxiliary_targets(data.train),
            validation_data=add_auxiliary_targets(data.validation),
            epochs=epochs,
            verbose=1,
        )
        best_epoch = int(
            pd.Series(history.history["val_stress_probabilities_macro_f1"]).idxmax()
        )
        results.append({
            **config,
            "best_epoch": best_epoch + 1,
            "validation_accuracy": float(
                history.history["val_stress_probabilities_accuracy"][best_epoch]
            ),
            "validation_macro_f1": float(
                history.history["val_stress_probabilities_macro_f1"][best_epoch]
            ),
            "validation_loss": float(
                history.history["val_stress_probabilities_loss"][best_epoch]
            ),
            "training_seconds": round(time.perf_counter() - started, 2),
        })
    frame = pd.DataFrame(results).sort_values(
        ["validation_macro_f1", "validation_accuracy"], ascending=False
    ).reset_index(drop=True)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    frame.to_csv(REPORTS_DIR / "tuning_results.csv", index=False)
    print("Best configuration:")
    print(json.dumps(frame.iloc[0].to_dict(), indent=2))
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    tune(args.trials, args.epochs, args.batch_size)
