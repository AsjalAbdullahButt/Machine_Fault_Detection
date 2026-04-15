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
)

os.makedirs(PLOTS_DIR, exist_ok=True)


def build_rul_lstm(input_shape, units=LSTM_UNITS, dropout=DROPOUT_RATE):
    """LSTM regressor for Remaining Useful Life (steps to failure)."""
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
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    val_split: float = VAL_SPLIT,
):
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
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def plot_rul_history(history, out_path: str):
    history_dict = getattr(history, "history", {}) or {}
    if "loss" not in history_dict:
        return

    epochs = np.array(getattr(history, "epoch", list(range(len(history_dict["loss"])))), dtype=np.int32) + 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if len(epochs) == 1:
        axes[0].scatter(epochs, history_dict["loss"], label="Train Huber", s=50)
        if "val_loss" in history_dict:
            axes[0].scatter(epochs, history_dict["val_loss"], label="Val Huber", s=50, marker="x")
        axes[0].set_xlim(0.5, 1.5)
        axes[0].set_xticks([1])
    else:
        axes[0].plot(epochs, history_dict["loss"], label="Train Huber", linewidth=2)
        if "val_loss" in history_dict:
            axes[0].plot(epochs, history_dict["val_loss"], label="Val Huber", linewidth=2, linestyle="--")
    axes[0].set_title("RUL Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Huber Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if "mae" in history_dict:
        if len(epochs) == 1:
            axes[1].scatter(epochs, history_dict["mae"], label="Train MAE", s=50)
            if "val_mae" in history_dict:
                axes[1].scatter(epochs, history_dict["val_mae"], label="Val MAE", s=50, marker="x")
            axes[1].set_xlim(0.5, 1.5)
            axes[1].set_xticks([1])
        else:
            axes[1].plot(epochs, history_dict["mae"], label="Train MAE", linewidth=2)
            if "val_mae" in history_dict:
                axes[1].plot(epochs, history_dict["val_mae"], label="Val MAE", linewidth=2, linestyle="--")
        axes[1].set_title("RUL MAE over Epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE (steps)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [Saved] {out_path}")


def evaluate_rul_model(model, X_test, y_test):
    y_pred = model.predict(X_test, verbose=0).flatten()
    mae = float(np.mean(np.abs(y_test - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))

    print("\n" + "=" * 65)
    print("  RUL LSTM — EVALUATION RESULTS")
    print("=" * 65)
    print(f"  MAE  (steps): {mae:.4f}")
    print(f"  RMSE (steps): {rmse:.4f}")

    return {
        "mae": mae,
        "rmse": rmse,
        "y_pred": y_pred,
    }
