import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import BatchNormalization, Dense, Dropout, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical

from config import (
    PLOTS_DIR,
    LSTM_UNITS,
    DROPOUT_RATE,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    VAL_SPLIT,
    ES_PATIENCE,
    LR_PATIENCE,
    MIN_LR,
    BINARY_THRESHOLD,
    get_plot_path,
)

os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Architecture helpers ──────────────────────────────────────────────────────

def build_binary_lstm(input_shape, units=LSTM_UNITS, dropout=DROPOUT_RATE):
    """Two-layer stacked LSTM for binary fault classification.
    Output: sigmoid -> probability of fault.
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
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def build_multiclass_lstm(input_shape, num_classes, units=LSTM_UNITS, dropout=DROPOUT_RATE):
    """Two-layer stacked LSTM for multiclass failure-type classification.
    Output: softmax -> probability over num_classes failure types.
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
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    model,
    X_train,
    y_train,
    class_weights: dict = None,
    epochs: int         = EPOCHS,
    batch_size: int     = BATCH_SIZE,
    val_split: float    = VAL_SPLIT,
):
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=ES_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=LR_PATIENCE, min_lr=MIN_LR, verbose=1,
        ),
    ]
    cw = {int(k): float(v) for k, v in class_weights.items()} if class_weights else None
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        class_weight=cw,
        verbose=1,
    )
    return history


# ── Plots ─────────────────────────────────────────────────────────────────────

def _save_fig(path: str, fig=None):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    target = fig if fig is not None else plt
    target.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [Saved] {path}")


def plot_training_history(history, title: str, plot_key: str):
    path = get_plot_path(plot_key)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    history_dict = getattr(history, "history", {}) or {}
    loss_keys = ["loss", "val_loss"]
    acc_keys = ["accuracy", "val_accuracy"]
    alt_acc_keys = ["binary_accuracy", "val_binary_accuracy"]

    if all(key in history_dict for key in loss_keys):
        axes[0].plot(history_dict["loss"], label="Train Loss", linewidth=2)
        axes[0].plot(history_dict["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
        axes[0].set_title("Loss over Epochs")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "Loss history unavailable", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_axis_off()

    metric_keys = acc_keys if all(key in history_dict for key in acc_keys) else alt_acc_keys
    if all(key in history_dict for key in metric_keys):
        train_key, val_key = metric_keys
        axes[1].plot(history_dict[train_key], label="Train Acc", linewidth=2)
        axes[1].plot(history_dict[val_key], label="Val Acc", linewidth=2, linestyle="--")
        axes[1].set_title("Accuracy over Epochs")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "Accuracy history unavailable", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()

    fig.tight_layout()
    _save_fig(path, fig=fig)


def plot_confusion_matrix(y_true, y_pred, labels, title: str, plot_key: str):
    path = get_plot_path(plot_key)
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, colorbar=True)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_fig(path, fig=fig)


def plot_roc_binary(y_true, y_prob, plot_key: str):
    path = get_plot_path(plot_key)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    fig = plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random classifier")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Binary LSTM — ROC Curve", fontweight="bold")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    _save_fig(path, fig=fig)
    return roc_auc


def plot_roc_multiclass(y_true, y_prob, class_names, plot_key: str):
    path      = get_plot_path(plot_key)
    n_classes = len(class_names)
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))

    fig = plt.figure(figsize=(9, 6))
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr, tpr):.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Multiclass LSTM — One-vs-Rest ROC Curves", fontweight="bold")
    plt.legend(fontsize=8, loc="lower right"); plt.grid(alpha=0.3)
    _save_fig(path, fig=fig)


def plot_f1_bars(report_dict: dict, title: str, plot_key: str):
    path    = get_plot_path(plot_key)
    classes = [k for k in report_dict if k not in ("accuracy", "macro avg", "weighted avg")]
    f1s     = [report_dict[k]["f1-score"] for k in classes]

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.5), 4))
    colors  = ["#2ecc71" if s >= 0.8 else "#f39c12" if s >= 0.5 else "#e74c3c" for s in f1s]
    bars    = ax.bar(classes, f1s, color=colors, edgecolor="black", linewidth=0.6)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel("F1 Score")
    ax.set_title(title, fontweight="bold")
    ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(path, fig=fig)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_binary(model, X_test, y_test, threshold: float = BINARY_THRESHOLD):
    """Full binary evaluation: metrics + all four plots."""
    sep = "=" * 65
    print(f"\n{sep}\n  BINARY LSTM — EVALUATION RESULTS\n{sep}")

    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    acc      = accuracy_score(y_test, y_pred)
    prec     = precision_score(y_test, y_pred, zero_division=0)
    rec      = recall_score(y_test, y_pred, zero_division=0)
    f1       = f1_score(y_test, y_pred, zero_division=0)
    loss_val, _ = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n  Test Loss      : {loss_val:.4f}")
    print(f"  Accuracy       : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision      : {prec:.4f}")
    print(f"  Recall         : {rec:.4f}")
    print(f"  F1 Score       : {f1:.4f}")

    roc_auc = plot_roc_binary(y_test, y_prob, "binary_roc")
    print(f"  ROC-AUC        : {roc_auc:.4f}")

    print(f"\n  Classification Report (threshold={threshold}):")
    print(classification_report(y_test, y_pred, target_names=["No Fault", "Fault"], zero_division=0))

    report_dict = classification_report(
        y_test, y_pred, target_names=["No Fault", "Fault"],
        output_dict=True, zero_division=0,
    )

    plot_confusion_matrix(y_test, y_pred, ["No Fault", "Fault"],
                          "Binary LSTM — Confusion Matrix", "binary_cm")
    plot_f1_bars(report_dict, "Binary LSTM — F1 Score per Class", "binary_f1")
    print(sep)

    return {
        "accuracy": acc, "precision": prec,
        "recall": rec,   "f1": f1, "roc_auc": roc_auc,
        "y_pred": y_pred, "y_prob": y_prob,
    }


def evaluate_multiclass(model, X_test, y_test, class_names, num_classes):
    """Full multiclass evaluation: metrics + all four plots."""
    sep = "=" * 65
    print(f"\n{sep}\n  MULTICLASS LSTM — EVALUATION RESULTS\n{sep}")

    y_prob      = model.predict(X_test, verbose=0)
    y_pred      = np.argmax(y_prob, axis=1)
    y_test_ohe  = to_categorical(y_test, num_classes)
    loss_val, acc = model.evaluate(X_test, y_test_ohe, verbose=0)

    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  Test Loss (CE)     : {loss_val:.4f}")
    print(f"  Accuracy           : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Weighted Precision : {prec:.4f}")
    print(f"  Weighted Recall    : {rec:.4f}")
    print(f"  Weighted F1        : {f1:.4f}")

    present_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    present_names  = [class_names[i] for i in present_labels if i < len(class_names)]

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
          labels=present_labels, target_names=present_names, zero_division=0))

    report_dict = classification_report(
        y_test, y_pred, labels=present_labels,
        target_names=present_names, output_dict=True, zero_division=0,
    )

    plot_confusion_matrix(y_test, y_pred, present_names,
                          "Multiclass LSTM — Confusion Matrix", "multi_cm")
    plot_roc_multiclass(y_test, y_prob, class_names, "multi_roc")
    plot_f1_bars(report_dict, "Multiclass LSTM — F1 Score per Class", "multi_f1")
    print(sep)

    return {
        "accuracy": acc, "precision": prec,
        "recall": rec,   "f1": f1,
        "y_pred": y_pred, "y_prob": y_prob,
    }
