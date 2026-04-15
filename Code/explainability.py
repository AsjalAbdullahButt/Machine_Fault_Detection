import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keras.models import Model

from config import PLOTS_DIR

os.makedirs(PLOTS_DIR, exist_ok=True)


def _safe_name(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _extract_attention_scores(transformer_model, sample_sequence: np.ndarray):
    """Return attention scores shaped (heads, T, T) for one sample sequence."""
    pos_layer = transformer_model.get_layer("positional_encoding")
    attn_layer = transformer_model.get_layer("self_attention")

    pos_model = Model(inputs=transformer_model.input, outputs=pos_layer.output)
    x = pos_model.predict(sample_sequence[np.newaxis, ...], verbose=0)
    _, scores = attn_layer(x, x, return_attention_scores=True, training=False)
    return scores.numpy()[0]


def plot_attention_heatmap(model, sample_sequence, feature_names, class_name, out_path):
    """Plot a (time_steps x features) attention-informed heatmap.

    Attention scores in transformers are over timestep-to-timestep pairs. To map
    attention onto features, this function combines timestep attention importance
    with per-feature signal magnitude in the given sample.
    """
    scores = _extract_attention_scores(model, sample_sequence)

    timestep_importance = scores.mean(axis=(0, 1))
    timestep_importance = timestep_importance / (np.sum(timestep_importance) + 1e-8)

    feature_importance = np.mean(np.abs(sample_sequence), axis=0)
    feature_importance = feature_importance / (np.sum(feature_importance) + 1e-8)

    heatmap = np.outer(timestep_importance, feature_importance)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd")
    ax.set_title(f"Transformer Attention Heatmap — {class_name}", fontweight="bold")
    ax.set_xlabel("Features")
    ax.set_ylabel("Time Step")
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(sample_sequence.shape[0]))
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [Saved] {out_path}")


def plot_attention_per_fault_type(transformer_model, X_seq, y_seq, class_names, feature_names):
    """Create one transformer attention heatmap per non-zero failure type."""
    saved_paths = []
    for cls_idx, cls_name in enumerate(class_names):
        if cls_idx == 0:
            continue
        idx = np.where(y_seq == cls_idx)[0]
        if len(idx) == 0:
            continue
        sample = X_seq[idx[0]]
        out_path = os.path.join(PLOTS_DIR, f"attention_heatmap_{_safe_name(cls_name)}.png")
        plot_attention_heatmap(
            model=transformer_model,
            sample_sequence=sample,
            feature_names=feature_names,
            class_name=cls_name,
            out_path=out_path,
        )
        saved_paths.append(out_path)
    return saved_paths


def compute_permutation_importance_per_failure_type(
    multiclass_model,
    X_test,
    y_test,
    class_names,
    feature_names,
):
    """Permutation importance matrix: failure_type x feature based on accuracy drop."""
    rows = []

    for cls_idx, cls_name in enumerate(class_names):
        if cls_idx == 0:
            continue
        mask = y_test == cls_idx
        if np.sum(mask) < 4:
            continue

        X_cls = X_test[mask].copy()
        y_cls = y_test[mask]

        base_pred = np.argmax(multiclass_model.predict(X_cls, verbose=0), axis=1)
        base_acc = float(np.mean(base_pred == y_cls))

        row = {"Failure Type": cls_name}
        for feat_idx, feat_name in enumerate(feature_names):
            X_perm = X_cls.copy()
            shuffled = X_perm[:, :, feat_idx].copy()
            np.random.shuffle(shuffled)
            X_perm[:, :, feat_idx] = shuffled

            perm_pred = np.argmax(multiclass_model.predict(X_perm, verbose=0), axis=1)
            perm_acc = float(np.mean(perm_pred == y_cls))
            row[feat_name] = max(0.0, base_acc - perm_acc)

        rows.append(row)

    return pd.DataFrame(rows)


def plot_feature_contribution_grouped(df_importance: pd.DataFrame, feature_names, out_path: str):
    """Grouped bar chart: failure types x features."""
    if df_importance.empty:
        return

    x = np.arange(len(df_importance))
    width = 0.12 if len(feature_names) >= 6 else 0.16

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, feat in enumerate(feature_names):
        ax.bar(
            x + (i - (len(feature_names) - 1) / 2) * width,
            df_importance[feat].values,
            width,
            label=feat,
        )

    ax.set_title("Feature Contribution by Failure Type (Permutation Importance)", fontweight="bold")
    ax.set_xlabel("Failure Type")
    ax.set_ylabel("Accuracy Drop After Feature Shuffle")
    ax.set_xticks(x)
    ax.set_xticklabels(df_importance["Failure Type"].tolist(), rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [Saved] {out_path}")
