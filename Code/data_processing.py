import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from imblearn.over_sampling import SMOTE, RandomOverSampler

from config import FEATURE_COLS, FAILURE_TYPES, TIME_STEPS, TEST_SIZE, RANDOM_SEED, TS_CV_SPLITS


# ── Loading & Inspection ──────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lstrip("\ufeff") for c in df.columns]
    return df


def inspect_data(df: pd.DataFrame) -> None:
    sep = "=" * 65
    print(f"\n{sep}")
    print("  DATASET OVERVIEW")
    print(sep)
    print(f"  Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
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


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame):
    """Clean, encode, and impute the raw dataframe.

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataframe with encoded columns added.
    le : LabelEncoder
        Fitted encoder for failure type labels.
    """
    df = df.copy()

    drop_cols = ["UDI", "Product ID"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    type_map = {"L": 0, "M": 1, "H": 2}
    if "Type" in df.columns:
        df["Type"] = df["Type"].map(type_map).fillna(0).astype(int)
        feature_cols_ext = ["Type"] + FEATURE_COLS
    else:
        feature_cols_ext = FEATURE_COLS

    missing = df[feature_cols_ext].isnull().sum().sum()
    if missing > 0:
        imputer = SimpleImputer(strategy="median")
        df[feature_cols_ext] = imputer.fit_transform(df[feature_cols_ext])
        print(f"  Imputed {missing} missing values (median strategy).")
    else:
        print("  No missing values found.")

    le = LabelEncoder()
    le.fit(FAILURE_TYPES)
    _mapped = df["Failure Type"].apply(lambda x: x if x in FAILURE_TYPES else "No Failure").to_numpy()
    df["failure_label"] = np.asarray(le.transform(_mapped))

    return df, le


# ── Feature / Label Split ─────────────────────────────────────────────────────
def split_features_labels(df: pd.DataFrame):
    """Return X (features), y_binary, y_multi."""
    use_cols = (["Type"] + FEATURE_COLS) if "Type" in df.columns else FEATURE_COLS
    X        = np.asarray(df[use_cols], dtype=np.float32)
    y_binary = np.asarray(df["Target"],        dtype=np.int32)
    y_multi  = np.asarray(df["failure_label"], dtype=np.int32)
    return X, y_binary, y_multi


# ── Class Imbalance: SMOTE ────────────────────────────────────────────────────
def apply_smote(X: np.ndarray, y: np.ndarray, random_state: int = RANDOM_SEED):
    """Apply SMOTE to balance the minority fault class."""
    classes, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(classes.tolist(), counts.tolist()))
    print(f"\n  Before SMOTE — class distribution: {class_dist}")

    min_count = int(counts.min()) if len(counts) else 0
    if min_count < 2:
        print("  [WARN] Minority class too small for SMOTE; using RandomOverSampler fallback.")
        sampler = RandomOverSampler(random_state=random_state)
    else:
        k_neighbors = max(1, min(5, min_count - 1))
        sampler = SMOTE(random_state=random_state, k_neighbors=k_neighbors)

    X_res, y_res = sampler.fit_resample(X, y)  # type: ignore[assignment]
    print(f"  After  SMOTE — class distribution: {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


def apply_smote_sequences(X_seq: np.ndarray, y_seq: np.ndarray, random_state: int = RANDOM_SEED):
    """Apply SMOTE directly on sequence samples by flattening each sequence."""
    n_samples, time_steps, n_features = X_seq.shape
    X_flat = X_seq.reshape(n_samples, time_steps * n_features)
    X_res, y_res = apply_smote(X_flat, y_seq, random_state=random_state)
    X_res_arr = np.asarray(X_res)
    X_res = X_res_arr.reshape(X_res_arr.shape[0], time_steps, n_features).astype(np.float32)
    return X_res, y_res.astype(np.int32)


def compute_class_weights(y: np.ndarray) -> dict:
    """Compute class weights inversely proportional to class frequency."""
    classes, counts = np.unique(y, return_counts=True)
    total   = len(y)
    weights = {int(c): float(total / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}
    print(f"  Class weights: {weights}")
    return weights


# ── Scaling ───────────────────────────────────────────────────────────────────
def scale_features(X_train, X_test, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


# ── Sequence Creation ─────────────────────────────────────────────────────────
def create_sequences(X: np.ndarray, y: np.ndarray, time_steps: int = TIME_STEPS):
    """Slide a window of length `time_steps` over the data."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps + 1):
        Xs.append(X[i: i + time_steps])
        ys.append(y[i + time_steps - 1])
    return np.array(Xs, dtype=np.float32), np.array(ys)


def compute_rul_targets(y_binary: np.ndarray) -> np.ndarray:
    """Compute Remaining Useful Life (RUL) as steps to next failure event.

    Failure samples have RUL=0. Non-failure samples count down to the next fault.
    If no future failure exists in the split, RUL increases toward split end.
    """
    rul = np.zeros(len(y_binary), dtype=np.float32)
    next_failure_idx = -1

    for i in range(len(y_binary) - 1, -1, -1):
        if int(y_binary[i]) == 1:
            next_failure_idx = i
            rul[i] = 0.0
        elif next_failure_idx >= 0:
            rul[i] = float(next_failure_idx - i)
        else:
            rul[i] = float(len(y_binary) - 1 - i)

    return rul


# ── Full Pipeline ─────────────────────────────────────────────────────────────
def build_pipeline(
    csv_path: str,
    time_steps: int   = TIME_STEPS,
    test_size: float  = TEST_SIZE,
    use_smote: bool   = True,
    random_state: int = RANDOM_SEED,
) -> dict:
    """End-to-end data pipeline.

    Parameters
    ----------
    csv_path     : path to the raw CSV dataset
    time_steps   : sliding-window length fed into the LSTM
    test_size    : fraction of rows held out for evaluation
    use_smote    : whether to apply SMOTE on the training split
    random_state : global RNG seed for reproducibility

    Returns
    -------
    dict with keys
    ──────────────
    X_train_bin           (N, time_steps, n_features) binary training sequences
    y_train_bin           (N,) binary labels after SMOTE
    X_train_mul           (N, time_steps, n_features) multiclass training sequences
    y_train_mul           (N,) multiclass labels after SMOTE
    X_test                (M, time_steps, n_features) held-out test sequences
    y_test_bin            (M,) binary test labels
    y_test_mul            (M,) multiclass test labels
    scaler                fitted StandardScaler
    label_encoder         fitted LabelEncoder for failure types
    class_weights_binary  dict {class_index: weight}
    class_weights_multi   dict {class_index: weight}
    n_features            number of input features
    num_classes           number of failure-type classes
    time_steps            window length used (echoed for downstream use)
    """
    print("\n[DATA] Loading dataset ...")
    df = load_data(csv_path)
    inspect_data(df)

    print("\n[DATA] Preprocessing ...")
    df, le = preprocess(df)

    X, y_bin, y_mul = split_features_labels(df)
    n_features  = X.shape[1]
    num_classes = len(le.classes_)

    # Temporal split — no shuffle to preserve time ordering
    split_idx = int(len(X) * (1 - test_size))
    X_train_raw, X_test_raw = X[:split_idx],     X[split_idx:]
    y_train_bin, y_test_bin = y_bin[:split_idx], y_bin[split_idx:]
    y_train_mul, y_test_mul = y_mul[:split_idx], y_mul[split_idx:]

    print(f"\n[DATA] Train size: {len(X_train_raw):,}  |  Test size: {len(X_test_raw):,}")

    X_train_s, X_test_s, scaler = scale_features(X_train_raw, X_test_raw)

    print(f"\n[DATA] Creating sequences (time_steps={time_steps}) ...")
    X_train_seq_bin, y_train_seq_bin = create_sequences(X_train_s, y_train_bin, time_steps)
    X_train_seq_mul, y_train_seq_mul = create_sequences(X_train_s, y_train_mul, time_steps)
    _,               y_train_seq_rul = create_sequences(X_train_s, compute_rul_targets(y_train_bin), time_steps)
    X_test_seq,      y_test_seq_bin  = create_sequences(X_test_s, y_test_bin, time_steps)
    _,               y_test_seq_mul  = create_sequences(X_test_s, y_test_mul, time_steps)
    _,               y_test_seq_rul  = create_sequences(X_test_s, compute_rul_targets(y_test_bin), time_steps)

    X_train_seq_rul = X_train_seq_bin.copy()

    if use_smote:
        print("\n[DATA] Applying sequence-level SMOTE for binary target ...")
        X_train_seq_bin, y_train_seq_bin = apply_smote_sequences(
            X_train_seq_bin, y_train_seq_bin, random_state
        )
        print("\n[DATA] Applying sequence-level SMOTE for multiclass target ...")
        X_train_seq_mul, y_train_seq_mul = apply_smote_sequences(
            X_train_seq_mul, y_train_seq_mul, random_state
        )

    print("\n[DATA] Computing class weights ...")
    cw_bin = compute_class_weights(np.asarray(y_train_seq_bin))
    cw_mul = compute_class_weights(np.asarray(y_train_seq_mul))

    print(f"  Binary train seq  : {X_train_seq_bin.shape}  labels: {y_train_seq_bin.shape}")
    print(f"  Multi  train seq  : {X_train_seq_mul.shape}  labels: {y_train_seq_mul.shape}")
    print(f"  Test   seq        : {X_test_seq.shape}       labels: {y_test_seq_bin.shape}")

    return {
        "X_train_bin":          X_train_seq_bin,
        "y_train_bin":          y_train_seq_bin,
        "X_train_mul":          X_train_seq_mul,
        "y_train_mul":          y_train_seq_mul,
        "X_train_rul":          X_train_seq_rul,
        "y_train_rul":          y_train_seq_rul.astype(np.float32),
        "X_test":               X_test_seq,
        "y_test_bin":           y_test_seq_bin,
        "y_test_mul":           y_test_seq_mul,
        "y_test_rul":           y_test_seq_rul.astype(np.float32),
        "scaler":               scaler,
        "label_encoder":        le,
        "class_weights_binary": cw_bin,
        "class_weights_multi":  cw_mul,
        "n_features":           n_features,
        "num_classes":          num_classes,
        "time_steps":           time_steps,
    }


def _prepare_fold_sequences(
    X: np.ndarray,
    y_bin: np.ndarray,
    y_mul: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    time_steps: int,
    use_smote: bool,
    random_state: int,
):
    X_train_raw, X_val_raw = X[train_idx], X[val_idx]
    y_train_bin, y_val_bin = y_bin[train_idx], y_bin[val_idx]
    y_train_mul, y_val_mul = y_mul[train_idx], y_mul[val_idx]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_raw)
    X_val_s = scaler.transform(X_val_raw)

    X_train_seq_bin, y_train_seq_bin = create_sequences(X_train_s, y_train_bin, time_steps)
    X_train_seq_mul, y_train_seq_mul = create_sequences(X_train_s, y_train_mul, time_steps)
    X_val_seq_bin, y_val_seq_bin = create_sequences(X_val_s, y_val_bin, time_steps)
    X_val_seq_mul, y_val_seq_mul = create_sequences(X_val_s, y_val_mul, time_steps)

    X_train_seq_rul, y_train_seq_rul = create_sequences(X_train_s, compute_rul_targets(y_train_bin), time_steps)
    X_val_seq_rul, y_val_seq_rul = create_sequences(X_val_s, compute_rul_targets(y_val_bin), time_steps)

    if use_smote:
        X_train_seq_bin, y_train_seq_bin = apply_smote_sequences(X_train_seq_bin, y_train_seq_bin, random_state)
        X_train_seq_mul, y_train_seq_mul = apply_smote_sequences(X_train_seq_mul, y_train_seq_mul, random_state)

    cw_bin = compute_class_weights(np.asarray(y_train_seq_bin))
    cw_mul = compute_class_weights(np.asarray(y_train_seq_mul))

    return {
        "scaler": scaler,
        "X_train_bin": X_train_seq_bin,
        "y_train_bin": y_train_seq_bin,
        "X_train_mul": X_train_seq_mul,
        "y_train_mul": y_train_seq_mul,
        "X_train_rul": X_train_seq_rul,
        "y_train_rul": y_train_seq_rul.astype(np.float32),
        "X_val_bin": X_val_seq_bin,
        "y_val_bin": y_val_seq_bin,
        "X_val_mul": X_val_seq_mul,
        "y_val_mul": y_val_seq_mul,
        "X_val_rul": X_val_seq_rul,
        "y_val_rul": y_val_seq_rul.astype(np.float32),
        "class_weights_binary": cw_bin,
        "class_weights_multi": cw_mul,
    }


def build_time_series_pipeline(
    csv_path: str,
    time_steps: int = TIME_STEPS,
    n_splits: int = TS_CV_SPLITS,
    use_smote: bool = True,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Build a fold-aware pipeline using TimeSeriesSplit for validation and a final holdout."""
    print("\n[DATA] Loading dataset ...")
    df = load_data(csv_path)
    inspect_data(df)

    print("\n[DATA] Preprocessing ...")
    df, le = preprocess(df)

    X, y_bin, y_mul = split_features_labels(df)
    n_features = X.shape[1]
    num_classes = len(le.classes_)

    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_indices = list(splitter.split(X))
    if not fold_indices:
        raise ValueError("TimeSeriesSplit produced no folds; check input size and n_splits.")

    cv_folds = []
    for fold_id, (train_idx, val_idx) in enumerate(fold_indices, start=1):
        fold = _prepare_fold_sequences(
            X=X,
            y_bin=y_bin,
            y_mul=y_mul,
            train_idx=train_idx,
            val_idx=val_idx,
            time_steps=time_steps,
            use_smote=use_smote,
            random_state=random_state,
        )
        fold.update({"fold_id": fold_id, "train_idx": train_idx, "val_idx": val_idx})
        cv_folds.append(fold)

    final_fold = cv_folds[-1]

    print(f"\n[DATA] TimeSeriesSplit folds: {len(cv_folds)}")
    print(f"  Final fold train seq : {final_fold['X_train_bin'].shape}  labels: {final_fold['y_train_bin'].shape}")
    print(f"  Final fold val seq   : {final_fold['X_val_bin'].shape}  labels: {final_fold['y_val_bin'].shape}")

    return {
        "X_train_bin": final_fold["X_train_bin"],
        "y_train_bin": final_fold["y_train_bin"],
        "X_train_mul": final_fold["X_train_mul"],
        "y_train_mul": final_fold["y_train_mul"],
        "X_train_rul": final_fold["X_train_rul"],
        "y_train_rul": final_fold["y_train_rul"],
        "X_test": final_fold["X_val_bin"],
        "y_test_bin": final_fold["y_val_bin"],
        "y_test_mul": final_fold["y_val_mul"],
        "y_test_rul": final_fold["y_val_rul"],
        "scaler": final_fold["scaler"],
        "label_encoder": le,
        "class_weights_binary": final_fold["class_weights_binary"],
        "class_weights_multi": final_fold["class_weights_multi"],
        "n_features": n_features,
        "num_classes": num_classes,
        "time_steps": time_steps,
        "cv_folds": cv_folds,
        "X_full": X,
        "y_full_bin": y_bin,
        "y_full_mul": y_mul,
    }
