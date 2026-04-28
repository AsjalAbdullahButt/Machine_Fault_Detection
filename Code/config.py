# ── Paths ─────────────────────────────────────────────────────────────────────
import os as _os

DATASET_PATH = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "Dataset", "predictive_maintenance.csv")
)
PLOTS_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "plots")
)
MODELS_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "models")
)
RESULTS_CSV = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "results_summary.csv")
)
SIM_LOG_CSV = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "simulation_log.csv")
)
REPORT_PDF = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "Report.pdf")
)

BINARY_MODEL_FILE = "binary_lstm.keras"
MULTI_MODEL_FILE  = "multiclass_lstm.keras"

# ── Plot subfolders (created automatically by ensure_plot_dirs) ────────────────
PLOTS_SUBDIRS = {
    "lstm":           _os.path.join(PLOTS_DIR, "lstm"),
    "cnn_lstm":       _os.path.join(PLOTS_DIR, "cnn_lstm"),
    "transformer":    _os.path.join(PLOTS_DIR, "transformer"),
    "rul":            _os.path.join(PLOTS_DIR, "rul"),
    "explainability": _os.path.join(PLOTS_DIR, "explainability"),
    "simulator":      _os.path.join(PLOTS_DIR, "simulator"),
}


def ensure_plot_dirs() -> None:
    """Create PLOTS_DIR and all architecture subfolders if they don't exist."""
    _os.makedirs(PLOTS_DIR, exist_ok=True)
    for path in PLOTS_SUBDIRS.values():
        _os.makedirs(path, exist_ok=True)


# ── Data pipeline ─────────────────────────────────────────────────────────────
TIME_STEPS  = 10      # LSTM sequence (sliding window) length
TEST_SIZE   = 0.20    # fraction of data held out for evaluation
RANDOM_SEED = 42

# ── Features & labels ─────────────────────────────────────────────────────────
FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

FAILURE_TYPES = [
    "No Failure",
    "Heat Dissipation Failure",
    "Power Failure",
    "Overstrain Failure",
    "Tool Wear Failure",
    "Random Failures",
]

# ── Model architecture ────────────────────────────────────────────────────────
LSTM_UNITS    = 64
DROPOUT_RATE  = 0.3
LEARNING_RATE = 1e-3

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS        = 60
BATCH_SIZE    = 64
VAL_SPLIT     = 0.10
ES_PATIENCE   = 15    # EarlyStopping patience
ES_MIN_DELTA  = 1e-3  # minimum val_loss improvement to reset EarlyStopping
LR_PATIENCE   = 4     # ReduceLROnPlateau patience
MIN_LR        = 1e-6
TS_CV_SPLITS  = 5
TUNING_EPOCHS = 12
TUNING_BATCH_SIZE = 64
TUNING_GRID = {
    "lstm_units": [32, 64, 128],
    "dropout": [0.2, 0.3, 0.4],
    "learning_rate": [1e-3, 5e-4],
}

# ── Inference ─────────────────────────────────────────────────────────────────
BINARY_THRESHOLD = 0.40   # decision boundary (lower than 0.5 → higher fault recall)

# ── Simulator ─────────────────────────────────────────────────────────────────
SIM_STEPS     = 200
PHASE_SPLITS  = (0.25, 0.20, 0.35, 0.20)   # NORMAL / WARM-UP / STRESS / FAILURE-ZONE
NOISE_STD     = 0.3

# Fault-boundary values derived from dataset EDA (75th percentile in fault cases)
FAULT_THRESHOLDS = {
    "Air temperature [K]":      301.5,
    "Process temperature [K]":  310.5,
    "Rotational speed [rpm]":   1420,    # below threshold → overload risk
    "Torque [Nm]":               48.0,
    "Tool wear [min]":          180,
}

# ── Plot naming convention (relative paths under PLOTS_DIR) ───────────────────
# Each value is a path relative to PLOTS_DIR, including the subfolder.
PLOT_NAMES = {
    # ── LSTM Binary
    "binary_history":               _os.path.join("lstm", "binary_history.png"),
    "binary_cm":                    _os.path.join("lstm", "binary_confusion_matrix.png"),
    "binary_roc":                   _os.path.join("lstm", "binary_roc_curve.png"),
    "binary_f1":                    _os.path.join("lstm", "binary_f1_scores.png"),
    "binary_pr":                    _os.path.join("lstm", "binary_pr_curve.png"),
    "binary_calibration":           _os.path.join("lstm", "binary_calibration_curve.png"),
    # ── LSTM Multiclass
    "multi_history":                _os.path.join("lstm", "multiclass_history.png"),
    "multi_cm":                     _os.path.join("lstm", "multiclass_confusion_matrix.png"),
    "multi_roc":                    _os.path.join("lstm", "multiclass_roc_curves.png"),
    "multi_f1":                     _os.path.join("lstm", "multiclass_f1_scores.png"),
    "multi_pr":                     _os.path.join("lstm", "multiclass_pr_curve.png"),
    # ── CNN-LSTM Binary
    "cnn_lstm_binary_history":      _os.path.join("cnn_lstm", "binary_history.png"),
    "cnn_lstm_binary_cm":           _os.path.join("cnn_lstm", "binary_confusion_matrix.png"),
    "cnn_lstm_binary_roc":          _os.path.join("cnn_lstm", "binary_roc_curve.png"),
    "cnn_lstm_binary_f1":           _os.path.join("cnn_lstm", "binary_f1_scores.png"),
    "cnn_lstm_binary_pr":           _os.path.join("cnn_lstm", "binary_pr_curve.png"),
    "cnn_lstm_binary_calibration":  _os.path.join("cnn_lstm", "binary_calibration_curve.png"),
    # ── CNN-LSTM Multiclass
    "cnn_lstm_multi_history":       _os.path.join("cnn_lstm", "multiclass_history.png"),
    "cnn_lstm_multi_cm":            _os.path.join("cnn_lstm", "multiclass_confusion_matrix.png"),
    "cnn_lstm_multi_roc":           _os.path.join("cnn_lstm", "multiclass_roc_curve.png"),
    "cnn_lstm_multi_f1":            _os.path.join("cnn_lstm", "multiclass_f1_scores.png"),
    "cnn_lstm_multi_pr":            _os.path.join("cnn_lstm", "multiclass_pr_curve.png"),
    # ── Transformer Binary
    "transformer_binary_history":       _os.path.join("transformer", "binary_history.png"),
    "transformer_binary_cm":            _os.path.join("transformer", "binary_confusion_matrix.png"),
    "transformer_binary_roc":           _os.path.join("transformer", "binary_roc_curve.png"),
    "transformer_binary_f1":            _os.path.join("transformer", "binary_f1_scores.png"),
    "transformer_binary_pr":            _os.path.join("transformer", "binary_pr_curve.png"),
    "transformer_binary_calibration":   _os.path.join("transformer", "binary_calibration_curve.png"),
    # ── Transformer Multiclass
    "transformer_multi_history":    _os.path.join("transformer", "multiclass_history.png"),
    "transformer_multi_cm":         _os.path.join("transformer", "multiclass_confusion_matrix.png"),
    "transformer_multi_roc":        _os.path.join("transformer", "multiclass_roc_curve.png"),
    "transformer_multi_f1":         _os.path.join("transformer", "multiclass_f1_scores.png"),
    "transformer_multi_pr":         _os.path.join("transformer", "multiclass_pr_curve.png"),
    # ── RUL
    "rul_history":                  _os.path.join("rul", "training_history.png"),
    # ── Simulator
    "sim_timeline":                 _os.path.join("simulator", "risk_timeline.png"),
    # ── Explainability
    "feature_contribution":         _os.path.join("explainability", "feature_contribution_by_failure_type.png"),
}


def get_plot_path(key: str) -> str:
    """Return the absolute path for a named plot key."""
    if key not in PLOT_NAMES:
        raise KeyError(f"Unknown plot key: '{key}'. Available: {list(PLOT_NAMES.keys())}")
    rel = PLOT_NAMES[key]
    full = _os.path.join(PLOTS_DIR, rel)
    # Ensure the subfolder exists
    _os.makedirs(_os.path.dirname(full), exist_ok=True)
    return full


# ── Model path lookup ─────────────────────────────────────────────────────────
# Maps (arch, task) → filename in MODELS_DIR.
# Actual files on disk:
#   binary_lstm.keras, multiclass_lstm.keras
#   cnn_lstm_binary.keras, cnn_lstm_multiclass.keras
#   transformer_binary.keras, transformer_multiclass.keras
#   rul_lstm.keras

_MODEL_FILE_MAP = {
    ("lstm",        "binary"):      "binary_lstm.keras",
    ("lstm",        "multiclass"):  "multiclass_lstm.keras",
    ("lstm",        "rul"):         "rul_lstm.keras",
    ("cnn_lstm",    "binary"):      "cnn_lstm_binary.keras",
    ("cnn_lstm",    "multiclass"):  "cnn_lstm_multiclass.keras",
    ("transformer", "binary"):      "transformer_binary.keras",
    ("transformer", "multiclass"):  "transformer_multiclass.keras",
}


def get_model_path(arch: str, task: str) -> str:
    """Return the absolute path for a model file.

    Parameters
    ----------
    arch : 'lstm', 'cnn_lstm', 'transformer'
    task : 'binary', 'multiclass', 'rul'
    """
    key = (arch.lower().strip(), task.lower().strip())
    if key not in _MODEL_FILE_MAP:
        raise KeyError(
            f"Unknown (arch, task): {key}. Valid: {list(_MODEL_FILE_MAP.keys())}"
        )
    return _os.path.join(MODELS_DIR, _MODEL_FILE_MAP[key])
