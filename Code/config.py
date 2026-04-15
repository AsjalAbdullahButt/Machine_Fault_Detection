# ── Paths ─────────────────────────────────────────────────────────────────────
PLOTS_DIR   = "plots"
REPORT_PDF  = "Report.pdf"
SIM_LOG_CSV = "simulation_log.csv"
RESULTS_CSV = "results_summary.csv"
MODELS_DIR  = "models"
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
    # Binary model
    "binary_history": "01_Binary_LSTM_Training_History.png",
    "binary_cm":      "02_Binary_LSTM_Confusion_Matrix.png",
    "binary_roc":     "03_Binary_LSTM_ROC_Curve.png",
    "binary_f1":      "04_Binary_LSTM_F1_Scores.png",
    # Multiclass model
    "multi_history":  "05_Multiclass_LSTM_Training_History.png",
    "multi_cm":       "06_Multiclass_LSTM_Confusion_Matrix.png",
    "multi_roc":      "07_Multiclass_LSTM_ROC_Curves.png",
    "multi_f1":       "08_Multiclass_LSTM_F1_Scores.png",
    # Simulator
    "sim_timeline":   "09_Simulation_Risk_Timeline.png",
}


def get_plot_path(key: str) -> str:
    """Return the full relative path for a named plot."""
    import os
    return os.path.join(PLOTS_DIR, PLOT_NAMES[key])
