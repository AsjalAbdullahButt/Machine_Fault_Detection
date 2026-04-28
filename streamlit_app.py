"""
Predictive Fault Detection System — Streamlit Dashboard
Architectures: LSTM · CNN-LSTM · Transformer · RUL
"""

import streamlit as st
import os
import sys
import json
import traceback
from datetime import datetime
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Path Setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(BASE_DIR, "Code")
sys.path.insert(0, CODE_DIR)

import config
from config import (
    PLOTS_DIR, PLOTS_SUBDIRS, MODELS_DIR, RESULTS_CSV,
    SIM_LOG_CSV, DATASET_PATH, get_plot_path, ensure_plot_dirs,
)
from data_processing import load_data

ensure_plot_dirs()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Machine Fault Detection",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .stApp { background-color: #0d1117; color: #c9d1d9; }

    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #21262d;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 { color: #58a6ff; }

    .stTabs [data-baseweb="tab-list"] {
        background-color: #161b22;
        border-bottom: 1px solid #21262d;
        gap: 0px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #8b949e;
        border-radius: 0;
        padding: 12px 22px;
        font-size: 13px;
        font-weight: 500;
        letter-spacing: 0.3px;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom: 2px solid #1f6feb !important;
        background-color: transparent !important;
    }

    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        color: #e6edf3 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 22px !important;
    }

    .stButton > button {
        background-color: #1f6feb;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 13px;
        font-weight: 500;
        padding: 10px 20px;
        transition: background 0.2s ease;
    }
    .stButton > button:hover { background-color: #388bfd; border: none; }

    .stDataFrame { border: 1px solid #21262d; border-radius: 8px; overflow: hidden; }

    hr { border-color: #21262d; }

    code {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 4px;
        padding: 2px 6px;
        color: #79c0ff;
    }

    .status-ok {
        display: inline-block;
        background: #1f3d1f; color: #56d364;
        border: 1px solid #2ea043;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 11px; font-weight: 600;
    }
    .status-miss {
        display: inline-block;
        background: #3d1f1f; color: #f85149;
        border: 1px solid #da3633;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 11px; font-weight: 600;
    }

    .page-header {
        padding: 22px 0 14px 0;
        border-bottom: 1px solid #21262d;
        margin-bottom: 24px;
    }
    .page-header h1 { font-size: 22px; font-weight: 600; color: #e6edf3; margin: 0; }
    .page-header p  { font-size: 13px; color: #8b949e; margin: 6px 0 0 0; }

    .info-box {
        background-color: #0d2136;
        border: 1px solid #1f6feb;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 13px;
        color: #79c0ff;
    }

    .plot-caption {
        font-size: 11px; color: #8b949e;
        text-align: center; margin-top: 4px;
        font-family: 'IBM Plex Mono', monospace;
    }

    h1, h2, h3 { color: #e6edf3; }
    h4, h5, h6 { color: #c9d1d9; }
    label { color: #8b949e !important; font-size: 12px !important; }

    [data-testid="stExpander"] {
        background-color: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
    }

    .risk-green  { color: #56d364; font-weight: 700; }
    .risk-yellow { color: #e3b341; font-weight: 700; }
    .risk-red    { color: #f85149; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor":   "#21262d", "axes.labelcolor": "#8b949e",
    "xtick.color":      "#8b949e", "ytick.color":     "#8b949e",
    "text.color":       "#c9d1d9", "grid.color":       "#21262d",
    "grid.linestyle":   "--",      "grid.alpha":       0.5,
    "legend.facecolor": "#161b22", "legend.edgecolor": "#21262d",
    "legend.fontsize":  9,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   12, "axes.titleweight": "semibold",
    "axes.titlecolor":  "#e6edf3", "axes.labelsize":   10,
    "figure.dpi":       120,
})

ACCENT = "#1f6feb"; RED = "#f85149"; GREEN = "#56d364"
YELLOW = "#e3b341"; PURPLE = "#bc8cff"; TEAL = "#39d353"; ORANGE = "#f0883e"

# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in [("training_done", False), ("last_metrics", {}), ("sim_results_df", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    st.markdown("**Training Settings**")

    epoch_mode = st.radio(
        "Epoch Budget",
        ["Quick — 5 epochs", "Standard — 20 epochs", "Full — 50 epochs"],
        index=1,
    )
    EPOCHS = {"Quick — 5 epochs": 5, "Standard — 20 epochs": 20, "Full — 50 epochs": 50}[epoch_mode]

    batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
    time_steps = st.slider("Sequence Length (Time Steps)", 5, 30, 10, 5)

    st.markdown("---")
    st.markdown("**Options**")
    load_existing   = st.checkbox("Use Pre-trained Models", value=True)
    verbose_logging = st.checkbox("Show Full Error Traces", value=False)

    st.markdown("---")
    st.markdown("**Model Files**")

    _actual_paths = {
        "LSTM Binary":            os.path.join(MODELS_DIR, "binary_lstm.keras"),
        "LSTM Multiclass":        os.path.join(MODELS_DIR, "multiclass_lstm.keras"),
        "CNN-LSTM Binary":        os.path.join(MODELS_DIR, "cnn_lstm_binary.keras"),
        "CNN-LSTM Multiclass":    os.path.join(MODELS_DIR, "cnn_lstm_multiclass.keras"),
        "Transformer Binary":     os.path.join(MODELS_DIR, "transformer_binary.keras"),
        "Transformer Multiclass": os.path.join(MODELS_DIR, "transformer_multiclass.keras"),
        "RUL LSTM":               os.path.join(MODELS_DIR, "rul_lstm.keras"),
    }

    all_present = True
    for name, path in _actual_paths.items():
        exists = os.path.exists(path)
        if not exists:
            all_present = False
        st.markdown(f"{'✅' if exists else '❌'} `{name}`")

    if all_present:
        st.success("All models ready")
    else:
        st.warning("Some models missing — run training first")

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>⚙️ Machine Fault Detection System</h1>
    <p>Predictive maintenance — LSTM · CNN-LSTM · Transformer · RUL estimation</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_data, tab_train, tab_results, tab_expl, tab_sim = st.tabs([
    "📊  Dataset",
    "🏋️  Training",
    "📈  Results & Plots",
    "🔍  Explainability",
    "🤖  Simulator",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATASET
# ════════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("### Dataset Overview")
    try:
        df = load_data(DATASET_PATH)

        c1, c2, c3, c4, c5 = st.columns(5)
        fault_count   = int(df["Target"].sum()) if "Target" in df.columns else 0
        fault_pct     = fault_count / len(df) * 100 if len(df) else 0
        missing_total = int(df.isnull().sum().sum())

        c1.metric("Total Samples",   f"{len(df):,}")
        c2.metric("Feature Columns", f"{df.shape[1]}")
        c3.metric("Fault Events",    f"{fault_count:,}")
        c4.metric("Fault Rate",      f"{fault_pct:.2f}%")
        c5.metric("Missing Values",  str(missing_total) if missing_total > 0 else "None")

        st.markdown("---")
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("**Sample Records (first 12 rows)**")
            st.dataframe(df.head(12), use_container_width=True, height=320)

        with col_right:
            st.markdown("**Failure Type Distribution**")
            if "Failure Type" in df.columns:
                ft_counts = df["Failure Type"].value_counts()
                fig, ax = plt.subplots(figsize=(5, 3.8))
                clrs = [ACCENT, GREEN, ORANGE, PURPLE, TEAL, RED][:len(ft_counts)]
                bars = ax.barh(ft_counts.index, ft_counts.to_numpy(), color=clrs, height=0.6)
                ax.bar_label(bars, fmt="%d", padding=4, fontsize=9, color="#c9d1d9")
                ax.set_xlabel("Count"); ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
            else:
                st.info("'Failure Type' column not found.")

        st.markdown("---")
        st.markdown("**Sensor Feature Distributions — Normal vs Fault**")

        numeric_cols = [c for c in config.FEATURE_COLS if c in df.columns]
        if numeric_cols:
            n_cols = 3
            rows   = (len(numeric_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(rows, n_cols, figsize=(13, 3.5 * rows))
            axes = np.array(axes).flatten()

            for idx, col in enumerate(numeric_cols):
                ax = axes[idx]
                if "Target" in df.columns:
                    no_f = df.loc[df["Target"] == 0, col].dropna()
                    flt  = df.loc[df["Target"] == 1, col].dropna()
                    ax.hist(no_f, bins=40, color=GREEN, alpha=0.65, label="Normal", density=True)
                    ax.hist(flt,  bins=40, color=RED,   alpha=0.65, label="Fault",  density=True)
                    ax.legend(fontsize=8)
                else:
                    ax.hist(df[col].dropna(), bins=40, color=ACCENT, alpha=0.8, density=True)
                ax.set_title(col, fontsize=10)
                ax.set_ylabel("Density", fontsize=8); ax.grid(alpha=0.3)

            for idx in range(len(numeric_cols), len(axes)):
                axes[idx].set_visible(False)

            fig.suptitle("Sensor Feature Distributions — Normal vs Fault", fontsize=12, y=1.01)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        st.markdown("---")
        st.markdown("**Statistical Summary**")
        st.dataframe(df.describe().round(3), use_container_width=True)

        if missing_total > 0:
            st.warning(f"⚠️  {missing_total} missing values found — median imputation will be applied.")
        else:
            st.success("✅  No missing values — dataset is clean.")

    except FileNotFoundError:
        st.error(f"Dataset not found: `{DATASET_PATH}`")
        st.info("Place `predictive_maintenance.csv` in the `Dataset/` folder.")
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        if verbose_logging:
            st.code(traceback.format_exc())


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — TRAINING
# ════════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.markdown("### Training Pipeline")

    col_sel1, col_sel2, col_status = st.columns([2, 2, 2])
    with col_sel1:
        arch = st.selectbox("Architecture", ["LSTM", "CNN-LSTM", "Transformer"])
    with col_sel2:
        task = st.selectbox("Task", ["Binary Classification", "Multiclass Classification", "RUL Estimation"])
    with col_status:
        st.markdown("**Pipeline Mode**")
        if load_existing:
            st.markdown('<span class="status-ok">LOAD MODE — Pre-trained</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-miss">TRAIN MODE — From scratch</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Model Availability**")

    ARCH_MODEL_MAP = {
        "LSTM":        ("binary_lstm.keras",        "multiclass_lstm.keras"),
        "CNN-LSTM":    ("cnn_lstm_binary.keras",     "cnn_lstm_multiclass.keras"),
        "Transformer": ("transformer_binary.keras",  "transformer_multiclass.keras"),
    }

    g_cols = st.columns(3)
    for idx, (arch_name, (bin_file, mul_file)) in enumerate(ARCH_MODEL_MAP.items()):
        bin_ok = os.path.exists(os.path.join(MODELS_DIR, bin_file))
        mul_ok = os.path.exists(os.path.join(MODELS_DIR, mul_file))
        with g_cols[idx]:
            st.markdown(f"**{arch_name}**")
            b = f'<span class="status-ok">Binary ✓</span>' if bin_ok else f'<span class="status-miss">Binary ✗</span>'
            m = f'<span class="status-ok">Multiclass ✓</span>' if mul_ok else f'<span class="status-miss">Multiclass ✗</span>'
            st.markdown(f"{b} &nbsp; {m}", unsafe_allow_html=True)
            if idx == 0:
                rul_ok = os.path.exists(os.path.join(MODELS_DIR, "rul_lstm.keras"))
                r = f'<span class="status-ok">RUL ✓</span>' if rul_ok else f'<span class="status-miss">RUL ✗</span>'
                st.markdown(r, unsafe_allow_html=True)

    st.markdown("---")
    btn_col1, btn_col2, _ = st.columns([1, 1, 3])
    with btn_col1:
        run_training = st.button("🚀  Start Training", use_container_width=True)
    with btn_col2:
        view_history = st.button("📉  View History", use_container_width=True)

    # ── Training ──────────────────────────────────────────────────────────────
    if run_training:
        task_key = (
            "binary"     if "Binary"     in task else
            "multiclass" if "Multiclass" in task else
            "rul"
        )
        progress_bar = st.progress(0, text="Initialising...")
        log_box      = st.empty()

        try:
            log_box.info("Loading modules...")
            from data_processing import (
                preprocess, create_sequences, split_features_labels,
                apply_smote_sequences, scale_features, compute_class_weights,
            )
            from models import (
                build_binary_lstm, build_multiclass_lstm,
                build_binary_cnn_lstm, build_multiclass_cnn_lstm,
                build_binary_transformer, build_multiclass_transformer,
                train_model, evaluate_binary, evaluate_multiclass,
            )
            from advanced_models import build_rul_lstm, train_rul_model, evaluate_rul_model

            progress_bar.progress(10, text="Loading dataset...")
            df_raw   = load_data(DATASET_PATH)
            df_pp, le = preprocess(df_raw)
            X_all, y_bin_all, y_mul_all = split_features_labels(df_pp)

            split_idx = int(len(X_all) * 0.80)
            X_tr_raw, X_te_raw = X_all[:split_idx], X_all[split_idx:]
            y_tr_bin, y_te_bin = y_bin_all[:split_idx], y_bin_all[split_idx:]
            y_tr_mul, y_te_mul = y_mul_all[:split_idx], y_mul_all[split_idx:]
            X_tr_s, X_te_s, _ = scale_features(X_tr_raw, X_te_raw)

            progress_bar.progress(25, text="Creating sequences...")
            y_tr = y_tr_bin if task_key in ("binary", "rul") else y_tr_mul
            y_te = y_te_bin if task_key in ("binary", "rul") else y_te_mul
            X_train_seq, y_train_seq = create_sequences(X_tr_s, np.asarray(y_tr, dtype=np.int32), time_steps)
            X_test_seq,  y_test_seq  = create_sequences(X_te_s, np.asarray(y_te, dtype=np.int32), time_steps)

            input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
            num_classes = int(len(np.unique(np.asarray(y_mul_all))))

            if task_key in ("binary", "multiclass"):
                progress_bar.progress(35, text="Applying SMOTE oversampling...")
                X_train_seq, y_train_seq = apply_smote_sequences(X_train_seq, y_train_seq)

            class_weights = compute_class_weights(np.asarray(y_train_seq)) if task_key in ("binary", "multiclass") else None  # type: ignore[arg-type]

            progress_bar.progress(45, text=f"Building {arch} model...")
            model_builders = {
                ("LSTM",        "binary"):     lambda: build_binary_lstm(input_shape),
                ("LSTM",        "multiclass"): lambda: build_multiclass_lstm(input_shape, num_classes),
                ("CNN-LSTM",    "binary"):     lambda: build_binary_cnn_lstm(input_shape),
                ("CNN-LSTM",    "multiclass"): lambda: build_multiclass_cnn_lstm(input_shape, num_classes),
                ("Transformer", "binary"):     lambda: build_binary_transformer(input_shape),
                ("Transformer", "multiclass"): lambda: build_multiclass_transformer(input_shape, num_classes),
            }

            if task_key == "rul":
                model = build_rul_lstm(input_shape)
            else:
                build_fn = model_builders.get((arch, task_key))
                if build_fn is None:
                    st.error(f"Unsupported combination: {arch} + {task_key}")
                    st.stop()
                model = build_fn()

            fname_map = {
                ("LSTM",        "binary"):     "binary_lstm.keras",
                ("LSTM",        "multiclass"): "multiclass_lstm.keras",
                ("CNN-LSTM",    "binary"):     "cnn_lstm_binary.keras",
                ("CNN-LSTM",    "multiclass"): "cnn_lstm_multiclass.keras",
                ("Transformer", "binary"):     "transformer_binary.keras",
                ("Transformer", "multiclass"): "transformer_multiclass.keras",
                ("LSTM",        "rul"):        "rul_lstm.keras",
            }
            os.makedirs(MODELS_DIR, exist_ok=True)
            save_path     = os.path.join(MODELS_DIR, fname_map.get((arch, task_key), f"{arch}_{task_key}.keras"))
            hist_json_path = save_path.replace(".keras", "_history.json")

            progress_bar.progress(55, text=f"Training {arch} — {task_key} ({EPOCHS} epochs max)...")

            if task_key == "rul":
                from data_processing import compute_rul_targets
                y_rul_tr = compute_rul_targets(np.asarray(y_tr_bin, dtype=np.int32))
                X_rul_seq, y_rul_seq = create_sequences(X_tr_s, y_rul_tr, time_steps)
                history = train_rul_model(model, X_rul_seq, y_rul_seq,
                                          epochs=EPOCHS, batch_size=batch_size,
                                          history_save_path=hist_json_path)
            else:
                history = train_model(model, X_train_seq, y_train_seq,
                                      class_weights=class_weights,
                                      epochs=EPOCHS, batch_size=batch_size,
                                      history_save_path=hist_json_path)

            progress_bar.progress(80, text="Saving model...")
            model.save(save_path)

            progress_bar.progress(90, text="Evaluating on held-out test set...")
            os.makedirs(PLOTS_DIR, exist_ok=True)

            if task_key == "binary":
                metrics = evaluate_binary(model, X_test_seq, y_test_seq,
                                          model_name=f"{arch} Binary")
                st.session_state.last_metrics  = metrics
                st.session_state.training_done = True
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                m2.metric("F1 Score", f"{metrics['f1']:.4f}")
                m3.metric("ROC-AUC",  f"{metrics['roc_auc']:.4f}")
                m4.metric("Recall",   f"{metrics['recall']:.4f}")

            elif task_key == "multiclass":
                class_names = list(le.classes_)
                metrics = evaluate_multiclass(model, X_test_seq, y_test_seq,
                                              class_names, num_classes,
                                              model_name=f"{arch} Multiclass")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                m2.metric("W-F1",     f"{metrics['f1']:.4f}")
                m3.metric("ROC-AUC",  f"{metrics['roc_auc']:.4f}")
                m4.metric("MCC",      f"{metrics['mcc']:.4f}")

            else:
                from data_processing import compute_rul_targets
                y_rul_te = compute_rul_targets(np.asarray(y_te_bin, dtype=np.int32))
                X_te_r, y_te_r = create_sequences(X_te_s, y_rul_te, time_steps)
                rul_m = evaluate_rul_model(model, X_te_r, y_te_r)
                m1, m2 = st.columns(2)
                m1.metric("Test MAE  (steps)", f"{rul_m['mae']:.2f}")
                m2.metric("Test RMSE (steps)", f"{rul_m['rmse']:.2f}")

            progress_bar.progress(100, text="Done!")
            log_box.success(f"✅  Training complete — model saved to `{save_path}`")

            # Show training curves inline
            h_dict = getattr(history, "history", {})
            if "loss" in h_dict:
                epochs_arr = list(range(1, len(h_dict["loss"]) + 1))
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                fig.suptitle(f"{arch} — {task} Training History", fontsize=12)
                axes[0].plot(epochs_arr, h_dict["loss"],   color=ACCENT, lw=2, label="Train Loss")
                if "val_loss" in h_dict:
                    axes[0].plot(epochs_arr, h_dict["val_loss"], color=ORANGE, lw=2, ls="--", label="Val Loss")
                axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid()

                acc_key = next((k for k in ["accuracy", "mae"] if k in h_dict), None)
                if acc_key:
                    axes[1].plot(epochs_arr, h_dict[acc_key], color=GREEN, lw=2, label=f"Train {acc_key}")
                    val_k = f"val_{acc_key}"
                    if val_k in h_dict:
                        axes[1].plot(epochs_arr, h_dict[val_k], color=PURPLE, lw=2, ls="--", label=f"Val {acc_key}")
                    axes[1].set_title(acc_key.upper()); axes[1].set_xlabel("Epoch")
                    axes[1].legend(); axes[1].grid()

                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

        except Exception as exc:
            log_box.error(f"Training failed: {exc}")
            if verbose_logging:
                st.code(traceback.format_exc())

    # ── View history ──────────────────────────────────────────────────────────
    if view_history:
        task_key = (
            "binary"     if "Binary"     in task else
            "multiclass" if "Multiclass" in task else
            "rul"
        )
        fname_map = {
            ("LSTM",        "binary"):     "binary_lstm_history.json",
            ("LSTM",        "multiclass"): "multiclass_lstm_history.json",
            ("CNN-LSTM",    "binary"):     "cnn_lstm_binary_history.json",
            ("CNN-LSTM",    "multiclass"): "cnn_lstm_multiclass_history.json",
            ("Transformer", "binary"):     "transformer_binary_history.json",
            ("Transformer", "multiclass"): "transformer_multiclass_history.json",
            ("LSTM",        "rul"):        "rul_lstm_history.json",
        }
        hist_file = fname_map.get((arch, task_key))
        hist_path = os.path.join(MODELS_DIR, hist_file) if hist_file else None

        if hist_path and os.path.exists(hist_path):
            try:
                from models import load_history_from_json
                hist = load_history_from_json(hist_path)
                if hist is None:
                    st.warning("History file exists but could not be parsed.")
                else:
                    h          = hist.history  # type: ignore[union-attr]
                    epochs_arr = list(range(1, len(h.get("loss", [])) + 1))

                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    fig.suptitle(f"{arch} — {task} Training History", fontsize=12)

                    if "loss" in h:
                        axes[0].plot(epochs_arr, h["loss"],   color=ACCENT, lw=2, label="Train Loss")
                        if "val_loss" in h:
                            axes[0].plot(epochs_arr, h["val_loss"], color=ORANGE, lw=2, ls="--", label="Val Loss")
                        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
                        axes[0].legend(); axes[0].grid()

                    acc_key = "accuracy" if "accuracy" in h else "mae"
                    if acc_key in h:
                        axes[1].plot(epochs_arr, h[acc_key], color=GREEN, lw=2, label=f"Train {acc_key}")
                        val_k = f"val_{acc_key}"
                        if val_k in h:
                            axes[1].plot(epochs_arr, h[val_k], color=PURPLE, lw=2, ls="--", label=f"Val {acc_key}")
                        axes[1].set_title(acc_key.upper()); axes[1].set_xlabel("Epoch")
                        axes[1].legend(); axes[1].grid()

                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
            except Exception as exc:
                st.error(f"Could not plot history: {exc}")
        else:
            st.warning(f"No history file found for `{arch} — {task}`. Train the model first.")

    st.markdown("---")
    st.markdown("**Quick-start via terminal (full pipeline)**")
    st.code(
        "cd Code\n"
        "python main.py --load-models --no-tune\n"
        "# Add --epochs 20 --batch 64 to override defaults",
        language="bash",
    )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTS & PLOTS
# ════════════════════════════════════════════════════════════════════════════════
with tab_results:
    st.markdown("### Model Performance Summary")

    if os.path.exists(RESULTS_CSV):
        try:
            results_df = pd.read_csv(RESULTS_CSV)
            num_cols   = [c for c in ["Accuracy", "F1", "ROC-AUC", "PR-AUC", "MCC", "Precision", "Recall"]
                          if c in results_df.columns]

            def _colour_best(col):
                if col.name not in num_cols:
                    return [""] * len(col)
                mx = col.max()
                return [
                    "background-color: #1f3d1f; color: #56d364; font-weight: 600;" if v == mx else ""
                    for v in col
                ]

            st.dataframe(results_df.style.apply(_colour_best), use_container_width=True, height=280)

            dl_c1, _ = st.columns([1, 4])
            with dl_c1:
                st.download_button(
                    "📥  Download CSV",
                    data=results_df.to_csv(index=False),
                    file_name="results_summary.csv",
                    mime="text/csv",
                )

            st.markdown("---")
            st.markdown("**Architecture Comparison — Key Metrics**")

            if "Architecture" in results_df.columns and "Task" in results_df.columns:
                metric_to_plot = st.selectbox(
                    "Select metric",
                    [m for m in ["F1", "ROC-AUC", "PR-AUC", "Accuracy", "MCC"] if m in results_df.columns],
                )
                fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
                palette   = [ACCENT, GREEN, ORANGE]

                for ax_idx, tname in enumerate(["Binary", "Multiclass"]):
                    sub = results_df[results_df["Task"] == tname].reset_index(drop=True)
                    if sub.empty:
                        axes[ax_idx].text(0.5, 0.5, "No data", ha="center", va="center",
                                          transform=axes[ax_idx].transAxes)
                        continue
                    bars = axes[ax_idx].bar(sub["Architecture"], sub[metric_to_plot],
                                            color=palette[:len(sub)], width=0.55, zorder=3)
                    axes[ax_idx].bar_label(bars, fmt="%.3f", padding=4, fontsize=10, color="#e6edf3")
                    axes[ax_idx].set_title(f"{tname} — {metric_to_plot}")
                    axes[ax_idx].set_ylim(0, 1.12)
                    axes[ax_idx].set_ylabel(metric_to_plot)
                    axes[ax_idx].grid(axis="y", alpha=0.4)
                    axes[ax_idx].set_xticklabels(sub["Architecture"], rotation=0)

                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

        except Exception as e:
            st.error(f"Could not load results: {e}")
    else:
        st.markdown("""
<div class="info-box">
    No <code>results_summary.csv</code> found yet. Run the full pipeline first:<br><br>
    <code>cd Code &amp;&amp; python main.py --load-models --no-tune</code>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Plot Gallery")

    # Build subfolder-aware index
    PLOT_GROUPS = {
        "All":            None,
        "LSTM":           PLOTS_SUBDIRS["lstm"],
        "CNN-LSTM":       PLOTS_SUBDIRS["cnn_lstm"],
        "Transformer":    PLOTS_SUBDIRS["transformer"],
        "RUL":            PLOTS_SUBDIRS["rul"],
        "Explainability": PLOTS_SUBDIRS["explainability"],
        "Simulator":      PLOTS_SUBDIRS["simulator"],
    }

    def _collect_pngs(folder: str) -> list[tuple[str, str]]:
        """Return list of (label, abs_path) for all .png files in folder."""
        if not os.path.isdir(folder):
            return []
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
        return [(f.replace("_", " ").replace(".png", "").title(), os.path.join(folder, f))
                for f in files]

    group_choice = st.radio("Filter by architecture", list(PLOT_GROUPS.keys()), horizontal=True)

    if group_choice == "All":
        # Collect from every subfolder
        all_plots: list[tuple[str, str, str]] = []  # (group, label, path)
        for gname, gdir in PLOT_GROUPS.items():
            if gdir is None:
                continue
            for label, path in _collect_pngs(gdir):
                all_plots.append((gname, label, path))
        filtered = [(f"{g} — {l}", p) for g, l, p in all_plots]
    else:
        gdir = PLOT_GROUPS[group_choice]
        filtered = _collect_pngs(gdir) if gdir else []

    if not filtered:
        st.markdown("""
<div class="info-box">
    No plots found for this filter. Run training or the full pipeline to generate them.
</div>
""", unsafe_allow_html=True)
    else:
        num_cols = 2
        for row_start in range(0, len(filtered), num_cols):
            batch = filtered[row_start: row_start + num_cols]
            cols  = st.columns(num_cols)
            for col_obj, (label, fpath) in zip(cols, batch):
                with col_obj:
                    st.markdown(f"**{label}**")
                    try:
                        st.image(fpath, use_container_width=True)
                    except Exception:
                        st.warning(f"Cannot load: {os.path.basename(fpath)}")
                    st.markdown(
                        f'<p class="plot-caption">{os.path.relpath(fpath, PLOTS_DIR)}</p>',
                        unsafe_allow_html=True,
                    )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════════════════
with tab_expl:
    st.markdown("### Explainability — Transformer Attention & Feature Importance")
    st.markdown(
        "**Attention heatmaps** show which time steps and features the Transformer focuses on "
        "for each failure type. **Permutation importance** reveals which sensor features cause "
        "the largest accuracy drop when shuffled, indicating their diagnostic importance."
    )
    st.markdown("---")

    expl_dir = PLOTS_SUBDIRS["explainability"]

    # ── Attention heatmaps ────────────────────────────────────────────────────
    st.markdown("#### Transformer Attention Heatmaps")
    st.caption("One heatmap per failure type — rows = time steps, columns = sensor features")

    attn_plots = [(f.replace("attention_heatmap_", "").replace("_", " ").replace(".png", "").title(),
                   os.path.join(expl_dir, f))
                  for f in sorted(os.listdir(expl_dir))
                  if f.startswith("attention_heatmap_") and f.endswith(".png")
                  ] if os.path.isdir(expl_dir) else []

    if attn_plots:
        n_cols = 2
        for row_start in range(0, len(attn_plots), n_cols):
            batch = attn_plots[row_start: row_start + n_cols]
            cols  = st.columns(n_cols)
            for col_obj, (label, fpath) in zip(cols, batch):
                with col_obj:
                    st.markdown(f"**{label}**")
                    st.image(fpath, use_container_width=True)
                    st.caption(
                        "Darker cells = higher combined attention × signal magnitude. "
                        "Rows are time steps; columns are input features."
                    )
    else:
        st.info("No attention heatmaps found. Run `python main.py --load-models --no-tune` with a Transformer model.")

    st.markdown("---")

    # ── Permutation importance ────────────────────────────────────────────────
    st.markdown("#### Feature Contribution by Failure Type (Permutation Importance)")
    st.caption("Bar height = accuracy drop when that feature is shuffled — higher is more important")

    feat_plot = os.path.join(expl_dir, "feature_contribution_by_failure_type.png")
    feat_csv  = os.path.join(expl_dir, "feature_contribution_by_failure_type.csv")

    if os.path.exists(feat_plot):
        st.image(feat_plot, use_container_width=True)
    else:
        st.info("Feature contribution plot not found. Run the full pipeline first.")

    if os.path.exists(feat_csv):
        st.markdown("**Numeric values (accuracy drop)**")
        try:
            fi_df = pd.read_csv(feat_csv)
            feat_cols = [c for c in fi_df.columns if c != "Failure Type"]

            def _heat_style(val):
                if not isinstance(val, float):
                    return ""
                intensity = min(int(val * 1000), 80)
                return f"background-color: rgba(31, 111, 235, {intensity/100:.2f}); color: #e6edf3;"

            styled_fi = fi_df.style.map(_heat_style, subset=tuple(feat_cols)).format(  # type: ignore[call-overload, arg-type]
                {c: "{:.4f}" for c in feat_cols}
            )
            st.dataframe(styled_fi, use_container_width=True)

            dl1, _ = st.columns([1, 4])
            with dl1:
                st.download_button(
                    "📥  Download CSV",
                    data=fi_df.to_csv(index=False),
                    file_name="feature_contribution.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.warning(f"Could not load feature contribution CSV: {e}")

    st.markdown("---")
    st.markdown("""
<div class="info-box">
<b>How to interpret these results:</b><br>
• <b>Attention heatmap</b>: The Transformer learns which sensor readings matter most per fault class.
  A bright cell at (t=8, Feature=Torque) means the model focuses heavily on Torque near the prediction window end.<br>
• <b>Permutation importance</b>: If shuffling <i>Tool wear</i> drops accuracy by 0.15 on Tool Wear Failure samples,
  that sensor is the primary diagnostic signal for that fault class.
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — SIMULATOR
# ════════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.markdown("### Robot Fault Simulator")
    st.markdown(
        "Simulates machine degradation across four phases — **Normal → Warm-up → Stress → Failure-Zone** "
        "— and tracks predicted fault risk and estimated Remaining Useful Life (RUL)."
    )
    st.markdown("---")

    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
        sim_steps = st.number_input("Simulation Steps", 20, 500, 200, 10)
    with s_col2:
        degradation_rate = st.slider(
            "Degradation Rate (drift scale)", 0.1, 3.0, 1.0, 0.1,
            help="Multiplies per-step sensor drift. 1.0 = nominal; 2.0 = twice as fast degradation.",
        )
    with s_col3:
        noise_level = st.slider(
            "Sensor Noise Std", 0.0, 1.0, 0.3, 0.05,
            help="Standard deviation of Gaussian noise added to each sensor reading per step.",
        )

    sim_uses_model = st.checkbox(
        "Use trained binary LSTM for risk scoring (requires models/ files)",
        value=False,
    )

    if sim_uses_model:
        st.info(
            "ℹ️  Model-based scoring loads the best-available binary model and the scaler from the "
            "dataset to compute fault probability at each step. Falls back to rule-based if any model "
            "file is missing."
        )

    st.markdown("---")

    if st.button("🚀  Run Simulation", use_container_width=False):
        sim_progress = st.progress(0, text="Starting simulation...")

        try:
            from simulator import RobotSimulator, score_risk

            sim = RobotSimulator(
                n_steps=int(sim_steps),
                noise_std=float(noise_level),
                drift_scale=float(degradation_rate),
            )

            # Attempt to load model + scaler for model-based scoring
            _binary_model = None
            _scaler       = None
            _using_model  = False

            if sim_uses_model:
                try:
                    from keras.models import load_model
                    from models import LearnedPositionalEncoding
                    from data_processing import preprocess, split_features_labels, scale_features

                    # Try each binary model in order of preference
                    for _mfile in ["transformer_binary.keras", "cnn_lstm_binary.keras", "binary_lstm.keras"]:
                        _mpath = os.path.join(MODELS_DIR, _mfile)
                        if os.path.exists(_mpath):
                            _binary_model = load_model(
                                _mpath,
                                custom_objects={"LearnedPositionalEncoding": LearnedPositionalEncoding},
                            )
                            break

                    if _binary_model is not None:
                        df_raw = load_data(DATASET_PATH)
                        df_pp, _ = preprocess(df_raw)
                        X_all, y_b, _ = split_features_labels(df_pp)
                        split_idx = int(len(X_all) * 0.80)
                        _, _, _scaler = scale_features(X_all[:split_idx], X_all[split_idx:])
                        _using_model = True
                        st.success(f"✅  Loaded model: `{_mfile}` for scoring")

                except Exception as _me:
                    st.warning(f"Could not load model ({_me}) — using rule-based scoring.")

            feat_cols = [
                "Air temperature [K]", "Process temperature [K]",
                "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
            ]
            all_feat_cols = ["Type"] + feat_cols  # includes Type for model input

            # Use the model's trained sequence length, not the sidebar slider value.
            # Mismatched shapes cause a predict() crash.
            _win_size = int(_binary_model.input_shape[1]) if _binary_model is not None else time_steps
            window  = deque(maxlen=_win_size)
            records = []

            for obs in sim:
                step  = obs["step"]
                phase = obs["phase"]

                if _using_model and _scaler is not None and _binary_model is not None:
                    raw_vec = np.array([[obs.get(f, 0) for f in all_feat_cols]], dtype=np.float32)
                    try:
                        scaled_vec = _scaler.transform(raw_vec)[0]
                    except Exception:
                        scaled_vec = raw_vec[0]
                    window.append(scaled_vec)

                    if len(window) >= _win_size:
                        seq       = np.array(list(window), dtype=np.float32)[np.newaxis, ...]
                        risk_prob = float(_binary_model.predict(seq, verbose=0)[0][0])
                    else:
                        phase_base = {"NORMAL": 0.05, "WARM-UP": 0.20, "STRESS": 0.55, "FAILURE-ZONE": 0.85}
                        risk_prob  = float(np.clip(phase_base.get(phase, 0.1) + np.random.normal(0, 0.03), 0, 1))
                else:
                    # Rule-based heuristic
                    phase_base = {"NORMAL": 0.05, "WARM-UP": 0.20, "STRESS": 0.55, "FAILURE-ZONE": 0.85}
                    risk_prob  = float(np.clip(phase_base.get(phase, 0.1) + np.random.normal(0, 0.04), 0, 1))

                risk_info = score_risk(risk_prob, obs)

                records.append({
                    "Step":            step,
                    "Phase":           phase,
                    "Risk (%)":        risk_info["risk_pct"],
                    "Tier":            risk_info["tier"],
                    "Scoring":         "Model" if _using_model else "Rule-based",
                    "Air Temp [K]":    round(obs["Air temperature [K]"], 2),
                    "Proc Temp [K]":   round(obs["Process temperature [K]"], 2),
                    "RPM":             round(obs["Rotational speed [rpm]"], 1),
                    "Torque [Nm]":     round(obs["Torque [Nm]"], 2),
                    "Tool Wear [min]": round(obs["Tool wear [min]"], 1),
                    "Warnings":        " | ".join(risk_info["warnings"]),
                    "Recommendation":  risk_info["recommendation"],
                })
                sim_progress.progress(step / int(sim_steps), text=f"Step {step}/{sim_steps} — {phase}")

            sim_df = pd.DataFrame(records)
            st.session_state.sim_results_df = sim_df
            sim_progress.progress(1.0, text="Simulation complete ✅")

        except Exception as exc:
            sim_progress.empty()
            st.error(f"Simulation failed: {exc}")
            if verbose_logging:
                st.code(traceback.format_exc())

    # ── Display results ───────────────────────────────────────────────────────
    sim_df = st.session_state.sim_results_df
    if sim_df is not None and len(sim_df) > 0:
        st.markdown("---")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Steps",      len(sim_df))
        k2.metric("Red Alert Steps",  int((sim_df["Tier"] == "RED").sum()))
        k3.metric("Yellow Steps",     int((sim_df["Tier"] == "YELLOW").sum()))
        k4.metric("Max Risk",         f"{sim_df['Risk (%)'].max():.1f}%")
        k5.metric("Avg Risk",         f"{sim_df['Risk (%)'].mean():.1f}%")

        if "Scoring" in sim_df.columns and sim_df["Scoring"].iloc[0] == "Model":
            st.success("🤖  Risk scores computed by the trained LSTM model")
        else:
            st.info("📐  Risk scores computed by rule-based phase heuristic")

        st.markdown("---")
        st.markdown("**Risk Timeline**")

        tier_color_map = {"GREEN": GREEN, "YELLOW": YELLOW, "RED": RED}

        fig = plt.figure(figsize=(13, 6))
        gs  = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1], hspace=0.08)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        for _, row in sim_df.iterrows():
            ax1.bar(row["Step"], row["Risk (%)"],
                    color=tier_color_map.get(row["Tier"], "#555"),
                    width=1.0, linewidth=0, alpha=0.85)

        prev_phase = None
        for _, row in sim_df.iterrows():
            if row["Phase"] != prev_phase:
                ax1.axvline(row["Step"], color="#444", lw=0.8, ls="--", alpha=0.7)
                ax1.text(row["Step"] + 0.5, 97, row["Phase"],
                         fontsize=8, color="#8b949e", va="top")
                prev_phase = row["Phase"]

        ax1.axhline(30, color=YELLOW, lw=1, ls=":", alpha=0.8, label="Caution (30%)")
        ax1.axhline(70, color=RED,    lw=1, ls=":", alpha=0.8, label="Critical (70%)")
        ax1.set_ylim(0, 108)
        ax1.set_ylabel("Fault Risk (%)", fontsize=10)
        ax1.set_title(
            f"Simulated Machine Fault Risk  |  Degradation ×{degradation_rate:.1f}  |  Noise σ={noise_level:.2f}",
            fontsize=11,
        )
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(axis="y", alpha=0.3)
        plt.setp(ax1.get_xticklabels(), visible=False)

        ax2.plot(sim_df["Step"], sim_df["Tool Wear [min]"], color=ORANGE, lw=1.8, label="Tool Wear [min]")
        ax2.set_ylabel("Tool Wear", fontsize=9)
        ax2.set_xlabel("Simulation Step", fontsize=10)
        ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── Sensor telemetry ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Sensor Readings Over Time**")

        sensor_map = {
            "Air Temp [K]":  (RED,    "Air Temperature [K]"),
            "Proc Temp [K]": (ORANGE, "Process Temperature [K]"),
            "RPM":           (ACCENT, "Rotational Speed [rpm]"),
            "Torque [Nm]":   (PURPLE, "Torque [Nm]"),
        }

        fig2, axes2 = plt.subplots(2, 2, figsize=(13, 5), sharex=True)
        axes2 = axes2.flatten()
        for i, (col, (clr, full_name)) in enumerate(sensor_map.items()):
            if col in sim_df.columns:
                axes2[i].plot(sim_df["Step"], sim_df[col], color=clr, lw=1.6)
                axes2[i].set_title(full_name, fontsize=10)
                axes2[i].set_ylabel(col, fontsize=8)
                axes2[i].grid(alpha=0.3)

        axes2[2].set_xlabel("Step"); axes2[3].set_xlabel("Step")
        fig2.suptitle("Sensor Telemetry — Degradation Profile", fontsize=11)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        # ── Risk tier breakdown ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Risk Tier Breakdown**")
        tier_counts = sim_df["Tier"].value_counts().reindex(["GREEN", "YELLOW", "RED"], fill_value=0)

        tier_cols = st.columns(3)
        for tc_col, (tname, cnt) in zip(tier_cols, tier_counts.items()):
            pct = cnt / len(sim_df) * 100
            color_cls = {"GREEN": "risk-green", "YELLOW": "risk-yellow", "RED": "risk-red"}[str(tname)]
            tc_col.markdown(
                f'<span class="{color_cls}">{tname}</span><br>'
                f'<b>{cnt} steps</b> ({pct:.1f}%)',
                unsafe_allow_html=True,
            )

        # ── Warnings summary ──────────────────────────────────────────────────
        if "Warnings" in sim_df.columns:
            warn_steps = sim_df[sim_df["Warnings"].str.len() > 0]
            if len(warn_steps):
                st.markdown("---")
                with st.expander(f"⚠️  Active Warnings ({len(warn_steps)} steps)", expanded=False):
                    all_warns = []
                    for _, row in warn_steps.iterrows():
                        for w in row["Warnings"].split(" | "):
                            if w.strip():
                                all_warns.append(w.strip())
                    from collections import Counter
                    warn_counts = Counter(all_warns)
                    for warn, cnt in warn_counts.most_common():
                        st.markdown(f"- **{warn}** — triggered {cnt}× across {len(warn_steps)} steps")

        # ── Data table ────────────────────────────────────────────────────────
        st.markdown("---")
        with st.expander("📋  View Raw Simulation Log", expanded=False):
            st.dataframe(sim_df, use_container_width=True, height=320)

        st.download_button(
            label="📥  Download Simulation Log",
            data=sim_df.to_csv(index=False),
            file_name=f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#8b949e; font-size:12px;'>"
    "Machine Fault Detection System &nbsp;·&nbsp; "
    "LSTM · CNN-LSTM · Transformer · RUL &nbsp;·&nbsp; "
    "TensorFlow / Keras &nbsp;·&nbsp; Scikit-learn"
    "</p>",
    unsafe_allow_html=True,
)
