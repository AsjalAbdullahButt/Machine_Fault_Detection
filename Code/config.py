# ── Paths ─────────────────────────────────────────────────────────────────────
import os as _os
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

# ── Plot naming convention ────────────────────────────────────────────────────
# Keys used internally; paths resolved at runtime via get_plot_path()
PLOT_NAMES = {
    # ── LSTM Binary
    "binary_history":            "lstm_binary_history.png",
    "binary_cm":                 "lstm_binary_confusion_matrix.png",
    "binary_roc":                "lstm_binary_roc_curve.png",
    "binary_f1":                 "lstm_binary_f1_scores.png",
    "binary_pr":                 "lstm_binary_pr_curve.png",
    "binary_calibration":        "lstm_binary_calibration_curve.png",
    # ── LSTM Multiclass
    "multi_history":             "lstm_multiclass_history.png",
    "multi_cm":                  "lstm_multiclass_confusion_matrix.png",
    "multi_roc":                 "lstm_multiclass_roc_curves.png",
    "multi_f1":                  "lstm_multiclass_f1_scores.png",
    "multi_pr":                  "lstm_multiclass_pr_curve.png",
    # ── CNN-LSTM Binary
    "cnn_lstm_binary_history":   "cnn_lstm_binary_history.png",
    "cnn_lstm_binary_cm":        "cnn_lstm_binary_confusion_matrix.png",
    "cnn_lstm_binary_roc":       "cnn_lstm_binary_roc_curve.png",
    "cnn_lstm_binary_f1":        "cnn_lstm_binary_f1_scores.png",
    "cnn_lstm_binary_pr":        "cnn_lstm_binary_pr_curve.png",
    "cnn_lstm_binary_calibration": "cnn_lstm_binary_calibration_curve.png",
    # ── CNN-LSTM Multiclass
    "cnn_lstm_multi_history":    "cnn_lstm_multiclass_history.png",
    "cnn_lstm_multi_cm":         "cnn_lstm_multiclass_confusion_matrix.png",
    "cnn_lstm_multi_roc":        "cnn_lstm_multiclass_roc_curve.png",
    "cnn_lstm_multi_f1":         "cnn_lstm_multiclass_f1_scores.png",
    "cnn_lstm_multi_pr":         "cnn_lstm_multiclass_pr_curve.png",
    # ── Transformer Binary
    "transformer_binary_history":     "transformer_binary_history.png",
    "transformer_binary_cm":          "transformer_binary_confusion_matrix.png",
    "transformer_binary_roc":         "transformer_binary_roc_curve.png",
    "transformer_binary_f1":          "transformer_binary_f1_scores.png",
    "transformer_binary_pr":          "transformer_binary_pr_curve.png",
    "transformer_binary_calibration": "transformer_binary_calibration_curve.png",
    # ── Transformer Multiclass
    "transformer_multi_history":  "transformer_multiclass_history.png",
    "transformer_multi_cm":       "transformer_multiclass_confusion_matrix.png",
    "transformer_multi_roc":      "transformer_multiclass_roc_curve.png",
    "transformer_multi_f1":       "transformer_multiclass_f1_scores.png",
    "transformer_multi_pr":       "transformer_multiclass_pr_curve.png",
    # ── RUL
    "rul_history":                "rul_training_history.png",
    # ── Simulator
    "sim_timeline":               "simulation_risk_timeline.png",
}


def get_plot_path(key: str) -> str:
    """Return the full path for a named plot."""
    return _os.path.join(PLOTS_DIR, PLOT_NAMES[key])
