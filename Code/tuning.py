from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import product

import numpy as np
from sklearn.metrics import f1_score

from config import TUNING_BATCH_SIZE, TUNING_EPOCHS, TUNING_GRID, BINARY_THRESHOLD
from models import build_binary_lstm, train_model


@dataclass
class TuningResult:
    units: int
    dropout: float
    learning_rate: float
    mean_f1: float
    std_f1: float
    fold_scores: list[float]

    def to_dict(self) -> dict:
        return {
            "lstm_units": self.units,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "mean_f1": self.mean_f1,
            "std_f1": self.std_f1,
            "fold_scores": self.fold_scores,
        }


def _evaluate_fold(model, X_val, y_val, threshold: float = BINARY_THRESHOLD) -> float:
    y_prob = model.predict(X_val, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    return float(f1_score(y_val, y_pred, zero_division=0))


def run_lstm_grid_search(
    cv_folds: list[dict],
    input_shape,
    output_path: str,
    epochs: int = TUNING_EPOCHS,
    batch_size: int = TUNING_BATCH_SIZE,
) -> TuningResult:
    """Manual grid search over LSTM hyperparameters using TimeSeriesSplit folds."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    best_result: TuningResult | None = None
    all_results: list[TuningResult] = []

    for units, dropout, learning_rate in product(
        TUNING_GRID["lstm_units"],
        TUNING_GRID["dropout"],
        TUNING_GRID["learning_rate"],
    ):
        fold_scores: list[float] = []
        print("\n" + "-" * 72)
        print(
            f"  TUNING CONFIG -> units={units}, dropout={dropout:.2f}, lr={learning_rate:g}"
        )
        print("-" * 72)

        for fold in cv_folds:
            model = build_binary_lstm(
                input_shape=input_shape,
                units=units,
                dropout=dropout,
                learning_rate=learning_rate,
            )
            train_model(
                model=model,
                X_train=fold["X_train_bin"],
                y_train=fold["y_train_bin"],
                class_weights=fold["class_weights_binary"],
                epochs=epochs,
                batch_size=batch_size,
            )
            fold_f1 = _evaluate_fold(model, fold["X_val_bin"], fold["y_val_bin"])
            fold_scores.append(fold_f1)
            print(f"    Fold {fold['fold_id']}: F1={fold_f1:.4f}")

        mean_f1 = float(np.mean(fold_scores))
        std_f1 = float(np.std(fold_scores))
        result = TuningResult(
            units=units,
            dropout=dropout,
            learning_rate=learning_rate,
            mean_f1=mean_f1,
            std_f1=std_f1,
            fold_scores=fold_scores,
        )
        all_results.append(result)

        print(f"  Mean F1 = {mean_f1:.4f} ± {std_f1:.4f}")

        if best_result is None or result.mean_f1 > best_result.mean_f1:
            best_result = result

    assert best_result is not None

    payload = {
        "best_config": best_result.to_dict(),
        "all_results": [r.to_dict() for r in all_results],
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print("\n" + "=" * 72)
    print("  BEST LSTM HYPERPARAMETERS")
    print("=" * 72)
    print(
        f"  units={best_result.units}, dropout={best_result.dropout:.2f}, lr={best_result.learning_rate:g}"
    )
    print(f"  Mean F1 = {best_result.mean_f1:.4f} ± {best_result.std_f1:.4f}")
    print(f"  Saved -> {output_path}")
    print("=" * 72)

    return best_result
