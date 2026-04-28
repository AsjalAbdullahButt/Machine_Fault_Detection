import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import BatchNormalization, Dense, Dropout, LSTM
from keras.losses import Huber
from keras.models import Sequential
from keras.optimizers import Adam

from config import (
    PLOTS_DIR,
    PLOTS_SUBDIRS,
    LSTM_UNITS,
    DROPOUT_RATE,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    VAL_SPLIT,
    ES_PATIENCE,
    ES_MIN_DELTA,
    LR_PATIENCE,
    MIN_LR,
    ensure_plot_dirs,
)

ensure_plot_dirs()

RUL_PLOT_DIR = PLOTS_SUBDIRS["rul"]


def build_rul_lstm(input_shape, units=LSTM_UNITS, dropout=DROPOUT_RATE):
    """LSTM regressor for Remaining Useful Life (steps-to-failure).

    Uses Huber loss (robust to outliers) and ReLU output to enforce non-negative predictions.
    """
    model = Sequential(
        [
            LSTM(units, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(dropout),
            LSTM(units // 2, return_sequences=False),
            BatchNormalization(),
            Dropout(dropout),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="relu"),
        ],
        name="rul_lstm",
    )
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=Huber(),
        metrics=["mae"],
    )
    model.summary()
    return model


def train_rul_model(
    model,
    X_train,
    y_train,
    epochs: int     = EPOCHS,
    batch_size: int = BATCH_SIZE,
    val_split: float = VAL_SPLIT,
    history_save_path: str = None,
):
    import json
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=ES_PATIENCE,
            min_delta=ES_MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=LR_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=1,
    )
    if history_save_path:
        os.makedirs(os.path.dirname(history_save_path) if os.path.dirname(history_save_path) else ".", exist_ok=True)
        with open(history_save_path, "w") as f:
            json.dump({
                "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
                "epoch":   history.epoch,
                "params":  history.params,
            }, f, indent=2)
        print(f"  [Saved] RUL History -> {history_save_path}")
    return history


def plot_rul_history(history, out_path: str):
    """Plot RUL training history (Huber loss + MAE) and save to rul/ subfolder."""
    history_dict = getattr(history, "history", {}) or {}
    if "loss" not in history_dict:
        return

    epochs = np.array(
        getattr(history, "epoch", list(range(len(history_dict["loss"])))),
        dtype=np.int32,
    ) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("RUL LSTM — Training History", fontsize=13, fontweight="bold")

    # Huber loss subplot
    if len(epochs) == 1:
        axes[0].scatter(epochs, history_dict["loss"], label="Train Huber", s=60)
        if "val_loss" in history_dict:
            axes[0].scatter(epochs, history_dict["val_loss"], label="Val Huber", s=60, marker="x")
        axes[0].set_xlim(0.5, 1.5)
        axes[0].set_xticks([1])
    else:
        axes[0].plot(epochs, history_dict["loss"], label="Train Huber", linewidth=2)
        if "val_loss" in history_dict:
            axes[0].plot(epochs, history_dict["val_loss"], label="Val Huber", linewidth=2, linestyle="--")
    axes[0].set_title("Huber Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Huber Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # MAE subplot
    if "mae" in history_dict:
        if len(epochs) == 1:
            axes[1].scatter(epochs, history_dict["mae"], label="Train MAE", s=60)
            if "val_mae" in history_dict:
                axes[1].scatter(epochs, history_dict["val_mae"], label="Val MAE", s=60, marker="x")
            axes[1].set_xlim(0.5, 1.5)
            axes[1].set_xticks([1])
        else:
            axes[1].plot(epochs, history_dict["mae"], label="Train MAE", linewidth=2)
            if "val_mae" in history_dict:
                axes[1].plot(epochs, history_dict["val_mae"], label="Val MAE", linewidth=2, linestyle="--")
        axes[1].set_title("MAE over Epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE (steps)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "MAE history unavailable", ha="center", va="center",
                     transform=axes[1].transAxes)
        axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [Saved] {out_path}")


def evaluate_rul_model(model, X_test, y_test) -> dict:
    """Evaluate RUL regression model and report MAE + RMSE."""
    y_pred = model.predict(X_test, verbose=0).flatten()
    mae    = float(np.mean(np.abs(y_test - y_pred)))
    rmse   = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

    print("\n" + "=" * 65)
    print("  RUL LSTM — EVALUATION RESULTS")
    print("=" * 65)
    print(f"  MAE  (steps to failure): {mae:.4f}")
    print(f"  RMSE (steps to failure): {rmse:.4f}")
    print("=" * 65)

    return {"mae": mae, "rmse": rmse, "y_pred": y_pred}
