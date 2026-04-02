"""
data_processing.py
==================
Handles all data loading, cleaning, feature engineering,
class-imbalance mitigation (SMOTE + class weights),
and sequence creation for LSTM time-series training.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# ── Constants ──────────────────────────────────────────────────────────────
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


# ── Loading & Inspection ────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Strip BOM if present
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    return df


def inspect_data(df: pd.DataFrame) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("  DATASET OVERVIEW")
    print(sep)
    print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\n  Columns: {list(df.columns)}")
    print(f"\n  First 5 rows:")
    print(df.head().to_string(index=False))
    print(f"\n  Summary statistics:")
    print(df.describe(include="all").round(3).to_string())
    print(f"\n  Target distribution:")
    print(df["Target"].value_counts().to_string())
    print(f"\n  Failure Type distribution:")
    print(df["Failure Type"].value_counts().to_string())
    print(sep)


# ── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean, encode, and impute the raw dataframe."""
    df = df.copy()

    # Drop ID columns that carry no predictive signal
    drop_cols = ["UDI", "Product ID"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Encode machine type (L/M/H) as ordinal integer
    type_map = {"L": 0, "M": 1, "H": 2}
    if "Type" in df.columns:
        df["Type"] = df["Type"].map(type_map).fillna(0).astype(int)
        FEATURE_COLS_EXTENDED = ["Type"] + FEATURE_COLS
    else:
        FEATURE_COLS_EXTENDED = FEATURE_COLS

    # Impute any missing numerics
    missing = df[FEATURE_COLS_EXTENDED].isnull().sum().sum()
    if missing > 0:
        imputer = SimpleImputer(strategy="median")
        df[FEATURE_COLS_EXTENDED] = imputer.fit_transform(df[FEATURE_COLS_EXTENDED])
        print(f"  Imputed {missing} missing values (median strategy).")
    else:
        print("  No missing values found.")

    # Encode Failure Type labels
    le = LabelEncoder()
    le.fit(FAILURE_TYPES)
    df["failure_label"] = le.transform(
        df["Failure Type"].apply(lambda x: x if x in FAILURE_TYPES else "No Failure")
    )

    return df, le


# ── Feature / Label Split ────────────────────────────────────────────────────
def split_features_labels(df: pd.DataFrame):
    """Return X (features), y_binary, y_multi."""
    use_cols = (["Type"] + FEATURE_COLS) if "Type" in df.columns else FEATURE_COLS
    X = df[use_cols].values.astype(np.float32)
    y_binary = df["Target"].values.astype(np.int32)
    y_multi = df["failure_label"].values.astype(np.int32)
    return X, y_binary, y_multi


# ── Class Imbalance: SMOTE ───────────────────────────────────────────────────
def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    """Apply SMOTE to balance the minority fault class."""
    print(f"\n  Before SMOTE — class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    sm = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"  After  SMOTE — class distribution: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


def compute_class_weights(y: np.ndarray) -> dict:
    """Compute class weights inversely proportional to class frequency."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {int(c): float(total / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}
    print(f"  Class weights: {weights}")
    return weights


# ── Scaling ──────────────────────────────────────────────────────────────────
def scale_features(X_train, X_test, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


# ── Sequence Creation ────────────────────────────────────────────────────────
def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int = 10):
    """Slide a window of length `time_steps` over the data."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps + 1):
        Xs.append(X[i: i + time_steps])
        ys.append(y[i + time_steps - 1])
    return np.array(Xs, dtype=np.float32), np.array(ys)


# ── Full Pipeline ─────────────────────────────────────────────────────────────
def build_pipeline(
    csv_path: str,
    time_steps: int = 10,
    test_size: float = 0.2,
    use_smote: bool = True,
    random_state: int = 42,
):
    """
    End-to-end data pipeline.

    Returns
    -------
    dict with keys:
        X_train_seq, y_train_binary, y_train_multi
        X_test_seq,  y_test_binary,  y_test_multi
        scaler, label_encoder, class_weights_binary, class_weights_multi
        n_features, num_classes
    """
    print("\n[DATA] Loading dataset …")
    df = load_data(csv_path)
    inspect_data(df)

    print("\n[DATA] Preprocessing …")
    df, le = preprocess(df)

    X, y_bin, y_mul = split_features_labels(df)
    n_features = X.shape[1]
    num_classes = len(le.classes_)

    # Train / test split (no shuffle — preserve temporal order)
    split_idx = int(len(X) * (1 - test_size))
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train_bin, y_test_bin = y_bin[:split_idx], y_bin[split_idx:]
    y_train_mul, y_test_mul = y_mul[:split_idx], y_mul[split_idx:]

    print(f"\n[DATA] Train size: {len(X_train_raw):,}  |  Test size: {len(X_test_raw):,}")

    # Scale
    X_train_s, X_test_s, scaler = scale_features(X_train_raw, X_test_raw)

    # SMOTE on flat (pre-sequence) training data
    if use_smote:
        print("\n[DATA] Applying SMOTE for binary target …")
        X_train_sm_bin, y_train_sm_bin = apply_smote(X_train_s, y_train_bin, random_state)

        print("\n[DATA] Applying SMOTE for multiclass target …")
        X_train_sm_mul, y_train_sm_mul = apply_smote(X_train_s, y_train_mul, random_state)
    else:
        X_train_sm_bin, y_train_sm_bin = X_train_s, y_train_bin
        X_train_sm_mul, y_train_sm_mul = X_train_s, y_train_mul

    # Class weights (computed on SMOTE'd data)
    print("\n[DATA] Computing class weights …")
    cw_bin = compute_class_weights(y_train_sm_bin)
    cw_mul = compute_class_weights(y_train_sm_mul)

    # Build sequences
    print(f"\n[DATA] Creating sequences (time_steps={time_steps}) …")
    X_train_seq_bin, y_train_seq_bin = create_sequences(X_train_sm_bin, y_train_sm_bin, time_steps)
    X_train_seq_mul, y_train_seq_mul = create_sequences(X_train_sm_mul, y_train_sm_mul, time_steps)
    X_test_seq, y_test_seq_bin = create_sequences(X_test_s, y_test_bin, time_steps)
    _, y_test_seq_mul = create_sequences(X_test_s, y_test_mul, time_steps)

    print(f"  Binary train seq  : {X_train_seq_bin.shape}  labels: {y_train_seq_bin.shape}")
    print(f"  Multi  train seq  : {X_train_seq_mul.shape}  labels: {y_train_seq_mul.shape}")
    print(f"  Test   seq        : {X_test_seq.shape}       labels: {y_test_seq_bin.shape}")

    return {
        # Binary
        "X_train_bin": X_train_seq_bin,
        "y_train_bin": y_train_seq_bin,
        # Multiclass
        "X_train_mul": X_train_seq_mul,
        "y_train_mul": y_train_seq_mul,
        # Test (shared X, two label sets)
        "X_test": X_test_seq,
        "y_test_bin": y_test_seq_bin,
        "y_test_mul": y_test_seq_mul,
        # Helpers
        "scaler": scaler,
        "label_encoder": le,
        "class_weights_binary": cw_bin,
        "class_weights_multi": cw_mul,
        "n_features": n_features,
        "num_classes": num_classes,
        "time_steps": time_steps,
    }
