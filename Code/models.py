"""
models.py
=========
Defines, trains, and evaluates both:
  1. Binary LSTM  — predicts Fault (1) vs No Fault (0)
  2. Multiclass LSTM — predicts one of 6 failure types
Includes: training plots, confusion matrix, ROC curve, full classification report.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe in all envs)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical


PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Architecture helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_binary_lstm(input_shape, units=64, dropout=0.3):
    """
    Two-layer stacked LSTM for binary fault classification.
    Output: sigmoid → probability of fault.
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
            Dense(1, activation="sigmoid"),
        ],
        name="binary_lstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def build_multiclass_lstm(input_shape, num_classes, units=64, dropout=0.3):
    """
    Two-layer stacked LSTM for multiclass failure-type classification.
    Output: softmax → probability over num_classes failure types.
    """
    model = Sequential(
        [
            LSTM(units, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(dropout),
            LSTM(units // 2, return_sequences=False),
            BatchNormalization(),
            Dropout(dropout),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(num_classes, activation="softmax"),
        ],
        name="multiclass_lstm",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    model,
    X_train,
    y_train,
    class_weights: dict = None,
    epochs: int = 60,
    batch_size: int = 64,
    val_split: float = 0.1,
):
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1
        ),
    ]

    # Convert class_weights dict keys to int (Keras requirement)
    cw = {int(k): float(v) for k, v in class_weights.items()} if class_weights else None

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        class_weight=cw,
        verbose=1,
    )
    return history


# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history, title: str, save_name: str):
    """Loss & accuracy curves for a training run."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Train Acc", linewidth=2)
    axes[1].plot(history.history["val_accuracy"], label="Val Acc", linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot saved] {path}")


def plot_confusion_matrix(y_true, y_pred, labels, title: str, save_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, colorbar=True)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot saved] {path}")


def plot_roc_binary(y_true, y_prob, save_name: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Binary LSTM — ROC Curve", fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot saved] {path}")
    return roc_auc


def plot_roc_multiclass(y_true, y_prob, class_names, save_name: str):
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(9, 6))
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass LSTM — One-vs-Rest ROC Curves", fontweight="bold")
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(alpha=0.3)
    path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot saved] {path}")


def plot_f1_bars(report_dict: dict, title: str, save_name: str):
    """Bar chart of per-class F1 scores."""
    classes = [k for k in report_dict if k not in ("accuracy", "macro avg", "weighted avg")]
    f1_scores = [report_dict[k]["f1-score"] for k in classes]

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.5), 4))
    colors = ["#2ecc71" if s >= 0.8 else "#f39c12" if s >= 0.5 else "#e74c3c" for s in f1_scores]
    bars = ax.bar(classes, f1_scores, color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score")
    ax.set_title(title, fontweight="bold")
    ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, save_name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_binary(model, X_test, y_test, threshold: float = 0.4):
    """
    Full binary evaluation: metrics + all plots.
    Uses threshold=0.4 (slightly lower than 0.5) to improve fault recall.
    """
    sep = "=" * 65
    print(f"\n{sep}")
    print("  BINARY LSTM — EVALUATION RESULTS")
    print(sep)

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    loss_val, _ = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n  Test Loss      : {loss_val:.4f}")
    print(f"  Accuracy       : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision      : {prec:.4f}")
    print(f"  Recall         : {rec:.4f}")
    print(f"  F1 Score       : {f1:.4f}")

    roc_auc = plot_roc_binary(y_test, y_prob, "binary_roc.png")
    print(f"  ROC-AUC        : {roc_auc:.4f}")

    print(f"\n  Classification Report (threshold={threshold}):")
    report_str = classification_report(
        y_test, y_pred, target_names=["No Fault", "Fault"], zero_division=0
    )
    print(report_str)

    report_dict = classification_report(
        y_test, y_pred, target_names=["No Fault", "Fault"],
        output_dict=True, zero_division=0
    )

    # Plots
    plot_confusion_matrix(
        y_test, y_pred,
        labels=["No Fault", "Fault"],
        title="Binary LSTM — Confusion Matrix",
        save_name="binary_cm.png",
    )
    plot_f1_bars(report_dict, "Binary LSTM — F1 per Class", "binary_f1.png")
    print(sep)

    return {
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "roc_auc": roc_auc,
        "y_pred": y_pred, "y_prob": y_prob,
    }


def evaluate_multiclass(model, X_test, y_test, class_names, num_classes):
    """
    Full multiclass evaluation: metrics + all plots.
    """
    sep = "=" * 65
    print(f"\n{sep}")
    print("  MULTICLASS LSTM — EVALUATION RESULTS")
    print(sep)

    y_prob = model.predict(X_test, verbose=0)          # shape (N, num_classes)
    y_pred = np.argmax(y_prob, axis=1)

    # One-hot labels for Keras evaluate
    y_test_ohe = to_categorical(y_test, num_classes)
    loss_val, acc = model.evaluate(X_test, y_test_ohe, verbose=0)

    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  Test Loss (CE)     : {loss_val:.4f}")
    print(f"  Accuracy           : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Weighted Precision : {prec:.4f}")
    print(f"  Weighted Recall    : {rec:.4f}")
    print(f"  Weighted F1        : {f1:.4f}")

    # Only use classes that appear in the test set
    present_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    present_names  = [class_names[i] for i in present_labels if i < len(class_names)]

    print(f"\n  Classification Report:")
    report_str = classification_report(
        y_test, y_pred,
        labels=present_labels, target_names=present_names,
        zero_division=0
    )
    print(report_str)

    report_dict = classification_report(
        y_test, y_pred,
        labels=present_labels, target_names=present_names,
        output_dict=True, zero_division=0
    )

    # Plots
    plot_confusion_matrix(
        y_test, y_pred,
        labels=present_names,
        title="Multiclass LSTM — Confusion Matrix",
        save_name="multiclass_cm.png",
    )
    plot_roc_multiclass(y_test, y_prob, class_names, "multiclass_roc.png")
    plot_f1_bars(report_dict, "Multiclass LSTM — F1 per Class", "multiclass_f1.png")
    print(sep)

    return {
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1,
        "y_pred": y_pred, "y_prob": y_prob,
    }
