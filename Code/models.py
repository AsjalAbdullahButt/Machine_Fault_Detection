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
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    matthews_corrcoef,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.saving import register_keras_serializable
from keras.layers import (
    Add,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Layer,
    LayerNormalization,
    LSTM,
    MaxPooling1D,
    MultiHeadAttention,
)
from keras.models import Model, Sequential
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
    BINARY_THRESHOLD,
    get_plot_path,
)

os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Architecture helpers ──────────────────────────────────────────────────────

def build_binary_lstm(input_shape, units=LSTM_UNITS, dropout=DROPOUT_RATE, learning_rate=LEARNING_RATE):
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
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def build_multiclass_lstm(input_shape, num_classes, units=LSTM_UNITS, dropout=DROPOUT_RATE, learning_rate=LEARNING_RATE):
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
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
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
    history_save_path: str = None,
):
    import json
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=ES_PATIENCE,
            min_delta=ES_MIN_DELTA,
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
    # Save history to JSON if path provided
    if history_save_path:
        os.makedirs(os.path.dirname(history_save_path) if os.path.dirname(history_save_path) else ".", exist_ok=True)
        with open(history_save_path, "w") as f:
            json.dump({
                "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
                "epoch": history.epoch,
                "params": history.params
            }, f, indent=2)
        print(f"  [Saved] History -> {history_save_path}")
    return history


def load_history_from_json(json_path: str):
    """Load a saved training history dict from JSON file."""
    import json
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        data = json.load(f)
    # Return a simple namespace that mimics Keras History object
    class _FakeHistory:
        pass
    h = _FakeHistory()
    h.history = data.get("history", {})
    h.epoch = data.get("epoch", list(range(len(next(iter(h.history.values()), [])))))
    h.params = data.get("params", {})
    return h


# ── Plots ─────────────────────────────────────────────────────────────────────

def _save_fig(path: str, fig=None):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    target = fig if fig is not None else plt
    target.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [Saved] {path}")


def _resolve_plot_path(plot_id: str) -> str:
    """Resolve either a configured plot key or a direct file path."""
    try:
        return get_plot_path(plot_id)
    except Exception:
        return plot_id


def plot_training_history(history, title: str, plot_key: str):
    path = _resolve_plot_path(plot_key)
    history_dict = getattr(history, "history", {}) or {}
    
    # CRITICAL: If history is empty or missing, attempt to load from JSON
    if not history_dict or "loss" not in history_dict:
        # Try to find and load corresponding JSON
        json_path = path.replace(".png", "_history.json").replace("plots/", "models/").replace("plots\\", "models\\")
        if os.path.exists(json_path):
            loaded = load_history_from_json(json_path)
            if loaded:
                history_dict = loaded.history
                history = loaded
        if not history_dict or "loss" not in history_dict:
            # Last resort: create a visible error plot
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f"Training history not available\nfor: {title}\n\nRun training first (not --load-models)", 
                   ha="center", va="center", fontsize=12, transform=ax.transAxes,
                   bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
            ax.set_axis_off()
            fig.suptitle(title, fontsize=13, fontweight="bold")
            _save_fig(path, fig=fig)
            return
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    loss_keys = ["loss", "val_loss"]
    acc_keys = ["accuracy", "val_accuracy"]
    alt_acc_keys = ["binary_accuracy", "val_binary_accuracy"]

    if all(key in history_dict for key in loss_keys):
        epochs = np.array(getattr(history, "epoch", list(range(len(history_dict["loss"])))), dtype=np.int32) + 1
        if len(epochs) == 1:
            axes[0].scatter(epochs, history_dict["loss"], label="Train Loss", s=50)
            axes[0].scatter(epochs, history_dict["val_loss"], label="Val Loss", s=50, marker="x")
            axes[0].set_xlim(0.5, 1.5)
            axes[0].set_xticks([1])
        else:
            axes[0].plot(epochs, history_dict["loss"], label="Train Loss", linewidth=2)
            axes[0].plot(epochs, history_dict["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
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
        epochs = np.array(getattr(history, "epoch", list(range(len(history_dict[train_key])))), dtype=np.int32) + 1
        if len(epochs) == 1:
            axes[1].scatter(epochs, history_dict[train_key], label="Train Acc", s=50)
            axes[1].scatter(epochs, history_dict[val_key], label="Val Acc", s=50, marker="x")
            axes[1].set_xlim(0.5, 1.5)
            axes[1].set_xticks([1])
        else:
            axes[1].plot(epochs, history_dict[train_key], label="Train Acc", linewidth=2)
            axes[1].plot(epochs, history_dict[val_key], label="Val Acc", linewidth=2, linestyle="--")
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
    path = _resolve_plot_path(plot_key)
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, colorbar=True)
    ax.set_title(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_fig(path, fig=fig)


def plot_roc_binary(y_true, y_prob, plot_key: str):
    path = _resolve_plot_path(plot_key)
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
    path      = _resolve_plot_path(plot_key)
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
    path    = _resolve_plot_path(plot_key)
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


def plot_pr_binary(y_true, y_prob, plot_key: str, title: str = "Binary — Precision-Recall Curve"):
    path = _resolve_plot_path(plot_key)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)

    fig = plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="#2c7fb8", lw=2, label=f"PR curve (AP = {ap_score:.3f})")
    baseline = np.mean(y_true)
    plt.hlines(y=baseline, xmin=0, xmax=1, colors="#7f8c8d", linestyles="--", label=f"Baseline={baseline:.3f}")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.02])
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(title, fontweight="bold")
    plt.legend(loc="lower left"); plt.grid(alpha=0.3)
    _save_fig(path, fig=fig)
    return ap_score


def plot_pr_multiclass(y_true, y_prob, class_names, plot_key: str, title: str = "Multiclass — One-vs-Rest PR Curves"):
    path = _resolve_plot_path(plot_key)
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig = plt.figure(figsize=(9, 6))
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        plt.plot(recall, precision, lw=2, label=f"{name} (AP={ap:.2f})")

    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.02])
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(title, fontweight="bold")
    plt.legend(fontsize=8, loc="lower left"); plt.grid(alpha=0.3)
    _save_fig(path, fig=fig)

    return average_precision_score(y_bin, y_prob, average="weighted")


def plot_calibration_binary(y_true, y_prob_raw, y_prob_used, plot_key: str, title: str = "Binary — Calibration Curve"):
    path = _resolve_plot_path(plot_key)
    frac_pos_raw, mean_pred_raw = calibration_curve(y_true, y_prob_raw, n_bins=10, strategy="quantile")
    frac_pos_used, mean_pred_used = calibration_curve(y_true, y_prob_used, n_bins=10, strategy="quantile")

    fig = plt.figure(figsize=(7, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    plt.plot(mean_pred_raw, frac_pos_raw, marker="o", label="Raw probabilities")
    if not np.allclose(y_prob_raw, y_prob_used):
        plt.plot(mean_pred_used, frac_pos_used, marker="s", label="Platt-scaled probabilities")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.0])
    plt.xlabel("Mean Predicted Probability"); plt.ylabel("Observed Fault Frequency")
    plt.title(title, fontweight="bold")
    plt.legend(loc="upper left"); plt.grid(alpha=0.3)
    _save_fig(path, fig=fig)

    return brier_score_loss(y_true, y_prob_used)


@register_keras_serializable(package="Custom")
class LearnedPositionalEncoding(Layer):
    """Learned positional encoding for short fixed-length sequences."""

    def build(self, input_shape):
        self.positional_embedding = self.add_weight(
            name="positional_embedding",
            shape=(1, int(input_shape[1]), int(input_shape[2])),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + self.positional_embedding

    def get_config(self):
        return super().get_config()


def build_binary_cnn_lstm(input_shape, units=LSTM_UNITS, dropout=DROPOUT_RATE, learning_rate=LEARNING_RATE):
    """CNN + LSTM hybrid for binary classification."""
    model = Sequential(
        [
            Conv1D(64, kernel_size=3, activation="relu", padding="same", input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            BatchNormalization(),
            LSTM(units, return_sequences=False),
            Dropout(dropout),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ],
        name="binary_cnn_lstm",
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def build_multiclass_cnn_lstm(input_shape, num_classes, units=LSTM_UNITS, dropout=DROPOUT_RATE, learning_rate=LEARNING_RATE):
    """CNN + LSTM hybrid for multiclass classification."""
    model = Sequential(
        [
            Conv1D(64, kernel_size=3, activation="relu", padding="same", input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            BatchNormalization(),
            LSTM(units, return_sequences=False),
            Dropout(dropout),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(num_classes, activation="softmax"),
        ],
        name="multiclass_cnn_lstm",
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def _build_transformer_backbone(input_shape, d_model: int = 64, heads: int = 2, dropout: float = DROPOUT_RATE):
    inputs = Input(shape=input_shape)
    x = Dense(d_model, name="input_projection")(inputs)
    x = LearnedPositionalEncoding(name="positional_encoding")(x)

    attn = MultiHeadAttention(
        num_heads=heads,
        key_dim=d_model // heads,
        dropout=dropout,
        name="self_attention",
    )(x, x)
    x = Add()([x, attn])
    x = LayerNormalization(epsilon=1e-6)(x)

    ffn = Dense(d_model * 2, activation="relu")(x)
    ffn = Dropout(dropout)(ffn)
    ffn = Dense(d_model)(ffn)
    x = Add()([x, ffn])
    x = LayerNormalization(epsilon=1e-6)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    return inputs, x


def build_binary_transformer(input_shape, dropout=DROPOUT_RATE, learning_rate=LEARNING_RATE):
    """Lightweight Transformer encoder for binary classification."""
    inputs, x = _build_transformer_backbone(input_shape=input_shape, d_model=64, heads=2, dropout=dropout)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs, name="binary_transformer")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def build_multiclass_transformer(input_shape, num_classes, dropout=DROPOUT_RATE, learning_rate=LEARNING_RATE):
    """Lightweight Transformer encoder for multiclass classification."""
    inputs, x = _build_transformer_backbone(input_shape=input_shape, d_model=64, heads=2, dropout=dropout)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs, name="multiclass_transformer")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_binary(
    model,
    X_test,
    y_test,
    threshold: float = BINARY_THRESHOLD,
    model_name: str = "Binary LSTM",
    plot_ids: dict | None = None,
    apply_platt_if_needed: bool = True,
):
    """Full binary evaluation: metrics + all four plots."""
    sep = "=" * 65
    print(f"\n{sep}\n  {model_name.upper()} — EVALUATION RESULTS\n{sep}")

    if plot_ids is None:
        plot_ids = {
            "roc": "binary_roc",
            "pr": os.path.join(PLOTS_DIR, "binary_pr_curve.png"),
            "calibration": os.path.join(PLOTS_DIR, "binary_calibration_curve.png"),
            "cm": "binary_cm",
            "f1": "binary_f1",
        }

    y_prob_raw = model.predict(X_test, verbose=0).flatten()

    prob_true_raw, prob_pred_raw = calibration_curve(y_test, y_prob_raw, n_bins=10, strategy="quantile")
    calibration_mae = float(np.mean(np.abs(prob_true_raw - prob_pred_raw)))
    platt_applied = False
    y_prob_used = y_prob_raw.copy()

    if apply_platt_if_needed and calibration_mae > 0.05 and len(np.unique(y_test)) > 1:
        platt = LogisticRegression(solver="lbfgs")
        platt.fit(y_prob_raw.reshape(-1, 1), y_test)
        y_prob_used = platt.predict_proba(y_prob_raw.reshape(-1, 1))[:, 1]
        platt_applied = True

    y_pred = (y_prob_used >= threshold).astype(int)

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

    roc_auc = plot_roc_binary(y_test, y_prob_used, plot_ids["roc"])
    pr_auc = plot_pr_binary(y_test, y_prob_used, plot_ids["pr"], title=f"{model_name} — Precision-Recall Curve")
    brier = plot_calibration_binary(
        y_test,
        y_prob_raw,
        y_prob_used,
        plot_ids["calibration"],
        title=f"{model_name} — Calibration Curve",
    )
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"  ROC-AUC        : {roc_auc:.4f}")
    print(f"  PR-AUC (AP)    : {pr_auc:.4f}")
    print(f"  MCC            : {mcc:.4f}")
    print(f"  Brier Score    : {brier:.4f}")
    print(f"  Calibration MAE: {calibration_mae:.4f}")
    print(f"  Platt Scaling  : {'Applied' if platt_applied else 'Not applied'}")

    print(f"\n  Classification Report (threshold={threshold}):")
    print(classification_report(y_test, y_pred, target_names=["No Fault", "Fault"], zero_division=0))

    report_dict = classification_report(
        y_test, y_pred, target_names=["No Fault", "Fault"],
        output_dict=True, zero_division=0,
    )

    plot_confusion_matrix(y_test, y_pred, ["No Fault", "Fault"],
                          f"{model_name} — Confusion Matrix", plot_ids["cm"])
    plot_f1_bars(report_dict, f"{model_name} — F1 Score per Class", plot_ids["f1"])
    print(sep)

    return {
        "accuracy": acc, "precision": prec,
        "recall": rec,   "f1": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc, "mcc": mcc,
        "brier": brier, "calibration_mae": calibration_mae,
        "platt_applied": platt_applied,
        "y_pred": y_pred, "y_prob": y_prob_used,
    }


def evaluate_multiclass(
    model,
    X_test,
    y_test,
    class_names,
    num_classes,
    model_name: str = "Multiclass LSTM",
    plot_ids: dict | None = None,
):
    """Full multiclass evaluation: metrics + all four plots."""
    sep = "=" * 65
    print(f"\n{sep}\n  {model_name.upper()} — EVALUATION RESULTS\n{sep}")

    if plot_ids is None:
        plot_ids = {
            "roc": "multi_roc",
            "pr": os.path.join(PLOTS_DIR, "multi_pr_curve.png"),
            "cm": "multi_cm",
            "f1": "multi_f1",
        }

    y_prob      = model.predict(X_test, verbose=0)
    y_pred      = np.argmax(y_prob, axis=1)
    loss_val, acc = model.evaluate(X_test, y_test, verbose=0)

    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    mcc  = matthews_corrcoef(y_test, y_pred)

    present_labels = sorted(np.unique(y_test))
    y_bin_present = label_binarize(y_test, classes=present_labels)
    y_prob_present = y_prob[:, present_labels]

    try:
        roc_auc = roc_auc_score(y_bin_present, y_prob_present, average="weighted", multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_bin_present, y_prob_present, average="weighted")
    except ValueError:
        pr_auc = float("nan")

    print(f"\n  Test Loss (CE)     : {loss_val:.4f}")
    print(f"  Accuracy           : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Weighted Precision : {prec:.4f}")
    print(f"  Weighted Recall    : {rec:.4f}")
    print(f"  Weighted F1        : {f1:.4f}")
    print(f"  ROC-AUC (OvR)      : {roc_auc:.4f}")
    print(f"  PR-AUC  (OvR)      : {pr_auc:.4f}")
    print(f"  MCC                : {mcc:.4f}")

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
                          f"{model_name} — Confusion Matrix", plot_ids["cm"])
    plot_roc_multiclass(y_test, y_prob, class_names, plot_ids["roc"])
    plot_pr_multiclass(y_test, y_prob, class_names, plot_ids["pr"], title=f"{model_name} — One-vs-Rest PR Curves")
    plot_f1_bars(report_dict, f"{model_name} — F1 Score per Class", plot_ids["f1"])
    print(sep)

    return {
        "accuracy": acc, "precision": prec,
        "recall": rec,   "f1": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc, "mcc": mcc,
        "y_pred": y_pred, "y_prob": y_prob,
    }
