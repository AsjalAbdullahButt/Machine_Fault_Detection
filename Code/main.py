import argparse
import os
import sys
import time
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from data_processing import build_time_series_pipeline
from advanced_models import build_rul_lstm, train_rul_model, evaluate_rul_model, plot_rul_history
from explainability import (
    plot_attention_per_fault_type,
    compute_permutation_importance_per_failure_type,
    plot_feature_contribution_grouped,
)
from models import (
    LearnedPositionalEncoding,
    build_binary_lstm,
    build_binary_cnn_lstm,
    build_binary_transformer,
    build_multiclass_lstm,
    build_multiclass_cnn_lstm,
    build_multiclass_transformer,
    evaluate_binary,
    evaluate_multiclass,
    plot_training_history,
    train_model,
)
from report_utils import save_pdf_report
from simulator import run_simulation
from keras.models import load_model
from tuning import run_lstm_grid_search

from config import (
    ensure_plot_dirs,
    PLOTS_DIR,
    PLOTS_SUBDIRS,
    REPORT_PDF,
    SIM_LOG_CSV,
    RESULTS_CSV,
    EPOCHS,
    BATCH_SIZE,
    TIME_STEPS,
    SIM_STEPS,
    MODELS_DIR,
    BINARY_MODEL_FILE,
    MULTI_MODEL_FILE,
    get_plot_path,
    FEATURE_COLS,
    TS_CV_SPLITS,
)

DEFAULT_CSV = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "Dataset", "predictive_maintenance.csv")
)
REPORT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "Report.pdf"))
MODELS_PATH = MODELS_DIR

ensure_plot_dirs()
os.makedirs(MODELS_PATH, exist_ok=True)

RUL_MODEL_PATH      = os.path.join(MODELS_PATH, "rul_lstm.keras")
BEST_LSTM_CFG_PATH  = os.path.join(MODELS_PATH, "best_lstm_config.json")


def load_project_model(path: str):
    return load_model(path, custom_objects={"LearnedPositionalEncoding": LearnedPositionalEncoding})


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predictive Fault Detection — LSTM pipeline"
    )
    parser.add_argument("--csv",          default=DEFAULT_CSV,  help="Path to dataset CSV")
    parser.add_argument("--epochs",       type=int, default=EPOCHS,     help="Max training epochs")
    parser.add_argument("--timesteps",    type=int, default=TIME_STEPS, help="LSTM sequence length")
    parser.add_argument("--batch",        type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--sim-steps",    type=int, default=SIM_STEPS,  help="Simulator steps")
    parser.add_argument("--no-sim",       action="store_true",           help="Skip robot simulator")
    parser.add_argument("--no-tune",      action="store_true",           help="Skip LSTM hyperparameter grid search")
    parser.add_argument("--load-models",  action="store_true",           help="Load pre-trained models, skip retraining")
    return parser.parse_args()


# ── Logging ───────────────────────────────────────────────────────────────────

def log_section(step: int, title: str):
    ts  = datetime.now().strftime("%H:%M:%S")
    bar = "=" * 65
    print(f"\n{bar}")
    print(f"  STEP {step} — {title}   [{ts}]")
    print(bar)


# ── Architecture registry ─────────────────────────────────────────────────────

ARCHITECTURES = [
    {
        "key": "lstm",
        "label": "LSTM",
        "binary_builder": build_binary_lstm,
        "multi_builder":  build_multiclass_lstm,
    },
    {
        "key": "cnn_lstm",
        "label": "CNN+LSTM",
        "binary_builder": build_binary_cnn_lstm,
        "multi_builder":  build_multiclass_cnn_lstm,
    },
    {
        "key": "transformer",
        "label": "Transformer",
        "binary_builder": build_binary_transformer,
        "multi_builder":  build_multiclass_transformer,
    },
]


def _model_paths(arch_key: str):
    if arch_key == "lstm":
        return (
            os.path.join(MODELS_PATH, BINARY_MODEL_FILE),
            os.path.join(MODELS_PATH, MULTI_MODEL_FILE),
        )
    return (
        os.path.join(MODELS_PATH, f"{arch_key}_binary.keras"),
        os.path.join(MODELS_PATH, f"{arch_key}_multiclass.keras"),
    )


def _plot_ids(arch_key: str, task: str) -> dict:
    """Return plot-id dict for a given architecture + task.

    Values are either:
      • a short config key  → resolved via get_plot_path()
      • an absolute path    → used directly (for plots not in PLOT_NAMES)
    """
    if arch_key == "lstm" and task == "binary":
        return {
            "history":     "binary_history",
            "cm":          "binary_cm",
            "roc":         "binary_roc",
            "pr":          "binary_pr",
            "calibration": "binary_calibration",
            "f1":          "binary_f1",
        }
    if arch_key == "lstm" and task == "multiclass":
        return {
            "history": "multi_history",
            "cm":      "multi_cm",
            "roc":     "multi_roc",
            "pr":      "multi_pr",
            "f1":      "multi_f1",
        }
    if arch_key == "cnn_lstm" and task == "binary":
        return {
            "history":     "cnn_lstm_binary_history",
            "cm":          "cnn_lstm_binary_cm",
            "roc":         "cnn_lstm_binary_roc",
            "pr":          "cnn_lstm_binary_pr",
            "calibration": "cnn_lstm_binary_calibration",
            "f1":          "cnn_lstm_binary_f1",
        }
    if arch_key == "cnn_lstm" and task == "multiclass":
        return {
            "history": "cnn_lstm_multi_history",
            "cm":      "cnn_lstm_multi_cm",
            "roc":     "cnn_lstm_multi_roc",
            "pr":      "cnn_lstm_multi_pr",
            "f1":      "cnn_lstm_multi_f1",
        }
    if arch_key == "transformer" and task == "binary":
        return {
            "history":     "transformer_binary_history",
            "cm":          "transformer_binary_cm",
            "roc":         "transformer_binary_roc",
            "pr":          "transformer_binary_pr",
            "calibration": "transformer_binary_calibration",
            "f1":          "transformer_binary_f1",
        }
    if arch_key == "transformer" and task == "multiclass":
        return {
            "history": "transformer_multi_history",
            "cm":      "transformer_multi_cm",
            "roc":     "transformer_multi_roc",
            "pr":      "transformer_multi_pr",
            "f1":      "transformer_multi_f1",
        }
    raise ValueError(f"Unknown arch/task combo: {arch_key}/{task}")


def save_results_summary(rows: list, out_path: str = RESULTS_CSV):
    df = pd.DataFrame(rows)
    df = df[
        [
            "Task", "Architecture", "Accuracy", "F1", "ROC-AUC",
            "PR-AUC", "MCC", "Training Time (s)", "Parameter Count",
            "Precision", "Recall",
        ]
    ]
    df.to_csv(out_path, index=False)

    sep = "=" * 85
    print(f"\n{sep}\n  MODEL COMPARISON TABLE (ALL ARCHITECTURES)\n{sep}")
    print(df.to_string(index=False))
    print(f"\n  Saved -> {out_path}")
    print(sep)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] Dataset not found: {args.csv}")
        print("  Place predictive_maintenance.csv in Dataset/ or pass --csv <path>")
        sys.exit(1)

    total_start = time.time()

    # ── Step 1: Data pipeline ─────────────────────────────────────────────────
    log_section(1, "DATA PIPELINE")
    pipeline = build_time_series_pipeline(
        csv_path=args.csv,
        time_steps=args.timesteps,
        n_splits=TS_CV_SPLITS,
        use_smote=True,
        random_state=42,
    )

    X_train_bin = pipeline["X_train_bin"]
    y_train_bin = pipeline["y_train_bin"]
    X_train_mul = pipeline["X_train_mul"]
    y_train_mul = pipeline["y_train_mul"]
    X_test      = pipeline["X_test"]
    y_test_bin  = pipeline["y_test_bin"]
    y_test_mul  = pipeline["y_test_mul"]
    X_train_rul = pipeline["X_train_rul"]
    y_train_rul = pipeline["y_train_rul"]
    y_test_rul  = pipeline["y_test_rul"]
    cv_folds    = pipeline["cv_folds"]
    scaler      = pipeline["scaler"]
    le          = pipeline["label_encoder"]
    cw_bin      = pipeline["class_weights_binary"]
    cw_mul      = pipeline["class_weights_multi"]
    n_features  = pipeline["n_features"]
    num_classes = pipeline["num_classes"]
    time_steps  = pipeline["time_steps"]

    class_names = list(le.classes_)
    input_shape = (time_steps, n_features)
    print(f"\n  Input shape : {input_shape}")
    print(f"  Num classes : {num_classes}  -> {class_names}")

    # ── Step 2: Optional LSTM hyperparameter tuning ───────────────────────────
    if not args.no_tune:
        log_section(2, "LSTM HYPERPARAMETER TUNING")
        best_lstm     = run_lstm_grid_search(cv_folds=cv_folds, input_shape=input_shape, output_path=BEST_LSTM_CFG_PATH)
        tuned_units   = best_lstm.units
        tuned_dropout = best_lstm.dropout
        tuned_lr      = best_lstm.learning_rate
    else:
        tuned_units = tuned_dropout = tuned_lr = None

    def _tuned(value, fallback):
        return fallback if value is None else value

    comparison_rows         = []
    binary_results_by_arch  = {}
    multi_results_by_arch   = {}
    trained_binary_models   = {}
    trained_multi_models    = {}

    step = 3 if not args.no_tune else 2
    for arch in ARCHITECTURES:
        arch_key   = arch["key"]
        arch_label = arch["label"]
        binary_ids = _plot_ids(arch_key, "binary")
        multi_ids  = _plot_ids(arch_key, "multiclass")
        bin_path, multi_path = _model_paths(arch_key)

        # ── Binary model ──────────────────────────────────────────────────────
        log_section(step, f"{arch_label} — BINARY TRAIN/EVAL")
        step += 1

        binary_train_time = 0.0
        if args.load_models and os.path.exists(bin_path):
            binary_model = load_project_model(bin_path)
            print(f"  [Loaded] {bin_path}")
        else:
            if arch_key in ("lstm", "cnn_lstm"):
                binary_model = arch["binary_builder"](
                    input_shape=input_shape,
                    units=_tuned(tuned_units, 64),
                    dropout=_tuned(tuned_dropout, 0.3),
                    learning_rate=_tuned(tuned_lr, 1e-3),
                )
            else:
                binary_model = arch["binary_builder"](
                    input_shape=input_shape,
                    dropout=_tuned(tuned_dropout, 0.3),
                    learning_rate=_tuned(tuned_lr, 1e-3),
                )
            print(f"\n  Training on {len(X_train_bin):,} sequences ...")
            t0 = time.time()
            history_bin = train_model(
                model=binary_model,
                X_train=X_train_bin,
                y_train=y_train_bin,
                class_weights=cw_bin,
                epochs=args.epochs,
                batch_size=args.batch,
                history_save_path=bin_path.replace(".keras", "_history.json"),
            )
            binary_train_time = time.time() - t0
            print(f"  Training time: {binary_train_time:.1f}s")
            plot_training_history(history_bin, f"{arch_label} — Binary Training History", binary_ids["history"])
            binary_model.save(bin_path)
            print(f"  [Saved] {bin_path}")

        binary_results = evaluate_binary(
            binary_model, X_test, y_test_bin,
            model_name=f"{arch_label} Binary",
            plot_ids=binary_ids,
            apply_platt_if_needed=True,
        )

        comparison_rows.append({
            "Task":              "Binary",
            "Architecture":      arch_label,
            "Accuracy":          round(binary_results["accuracy"], 4),
            "F1":                round(binary_results["f1"], 4),
            "ROC-AUC":           round(binary_results["roc_auc"], 4),
            "PR-AUC":            round(binary_results["pr_auc"], 4),
            "MCC":               round(binary_results["mcc"], 4),
            "Training Time (s)": round(binary_train_time, 2),
            "Parameter Count":   int(binary_model.count_params()),
            "Precision":         round(binary_results["precision"], 4),
            "Recall":            round(binary_results["recall"], 4),
        })
        binary_results_by_arch[arch_key] = binary_results
        trained_binary_models[arch_key]  = binary_model

        # ── Multiclass model ──────────────────────────────────────────────────
        log_section(step, f"{arch_label} — MULTICLASS TRAIN/EVAL")
        step += 1

        multi_train_time = 0.0
        if args.load_models and os.path.exists(multi_path):
            multi_model = load_project_model(multi_path)
            print(f"  [Loaded] {multi_path}")
        else:
            if arch_key in ("lstm", "cnn_lstm"):
                multi_model = arch["multi_builder"](
                    input_shape=input_shape,
                    num_classes=num_classes,
                    units=_tuned(tuned_units, 64),
                    dropout=_tuned(tuned_dropout, 0.3),
                    learning_rate=_tuned(tuned_lr, 1e-3),
                )
            else:
                multi_model = arch["multi_builder"](
                    input_shape=input_shape,
                    num_classes=num_classes,
                    dropout=_tuned(tuned_dropout, 0.3),
                    learning_rate=_tuned(tuned_lr, 1e-3),
                )
            print(f"\n  Training on {len(X_train_mul):,} sequences ...")
            t0 = time.time()
            history_mul = train_model(
                model=multi_model,
                X_train=X_train_mul,
                y_train=y_train_mul,
                class_weights=cw_mul,
                epochs=args.epochs,
                batch_size=args.batch,
                history_save_path=multi_path.replace(".keras", "_history.json"),
            )
            multi_train_time = time.time() - t0
            print(f"  Training time: {multi_train_time:.1f}s")
            plot_training_history(history_mul, f"{arch_label} — Multiclass Training History", multi_ids["history"])
            multi_model.save(multi_path)
            print(f"  [Saved] {multi_path}")

        multi_results = evaluate_multiclass(
            multi_model, X_test, y_test_mul, class_names, num_classes,
            model_name=f"{arch_label} Multiclass",
            plot_ids=multi_ids,
        )

        comparison_rows.append({
            "Task":              "Multiclass",
            "Architecture":      arch_label,
            "Accuracy":          round(multi_results["accuracy"], 4),
            "F1":                round(multi_results["f1"], 4),
            "ROC-AUC":           round(multi_results["roc_auc"], 4),
            "PR-AUC":            round(multi_results["pr_auc"], 4),
            "MCC":               round(multi_results["mcc"], 4),
            "Training Time (s)": round(multi_train_time, 2),
            "Parameter Count":   int(multi_model.count_params()),
            "Precision":         round(multi_results["precision"], 4),
            "Recall":            round(multi_results["recall"], 4),
        })
        multi_results_by_arch[arch_key] = multi_results
        trained_multi_models[arch_key]  = multi_model

    # ── Model comparison summary ───────────────────────────────────────────────
    log_section(step, "MODEL COMPARISON SUMMARY")
    step += 1
    save_results_summary(comparison_rows)

    # Pick best binary arch by PR-AUC for simulator and report
    best_arch         = max(binary_results_by_arch, key=lambda k: binary_results_by_arch[k]["pr_auc"])
    best_binary_res   = binary_results_by_arch[best_arch]
    best_binary_model = trained_binary_models[best_arch]
    best_multi_model  = trained_multi_models[best_arch]
    best_multi_res    = multi_results_by_arch[best_arch]
    print(f"\n  Best arch by PR-AUC: {best_arch} ({best_binary_res['pr_auc']:.4f})")

    # ── RUL regression model ───────────────────────────────────────────────────
    log_section(step, "RUL LSTM — TRAIN/EVALUATION")
    step += 1

    rul_train_time = 0.0
    if args.load_models and os.path.exists(RUL_MODEL_PATH):
        rul_model = load_project_model(RUL_MODEL_PATH)
        print(f"  [Loaded] {RUL_MODEL_PATH}")
    else:
        rul_model = build_rul_lstm(input_shape=input_shape)
        t0 = time.time()
        history_rul = train_rul_model(
            model=rul_model,
            X_train=X_train_rul,
            y_train=y_train_rul,
            epochs=args.epochs,
            batch_size=args.batch,
            history_save_path=RUL_MODEL_PATH.replace(".keras", "_history.json"),
        )
        rul_train_time = time.time() - t0
        print(f"  Training time: {rul_train_time:.1f}s")
        plot_rul_history(history_rul, get_plot_path("rul_history"))
        rul_model.save(RUL_MODEL_PATH)
        print(f"  [Saved] {RUL_MODEL_PATH}")

    rul_results = evaluate_rul_model(rul_model, X_test, y_test_rul)

    # ── Explainability ─────────────────────────────────────────────────────────
    transformer_binary = trained_binary_models.get("transformer")
    transformer_multi  = trained_multi_models.get("transformer")
    if transformer_binary is not None and transformer_multi is not None:
        log_section(step, "EXPLAINABILITY (TRANSFORMER ATTENTION + PERMUTATION IMPORTANCE)")
        step += 1

        feature_names = (["Type"] + FEATURE_COLS) if n_features == 6 else FEATURE_COLS

        attention_paths = plot_attention_per_fault_type(
            transformer_model=transformer_binary,
            X_seq=X_test,
            y_seq=y_test_mul,
            class_names=class_names,
            feature_names=feature_names,
        )

        importance_df = compute_permutation_importance_per_failure_type(
            multiclass_model=transformer_multi,
            X_test=X_test,
            y_test=y_test_mul,
            class_names=class_names,
            feature_names=feature_names,
        )
        importance_csv  = os.path.join(PLOTS_SUBDIRS["explainability"], "feature_contribution_by_failure_type.csv")
        importance_plot = get_plot_path("feature_contribution")
        importance_df.to_csv(importance_csv, index=False)
        plot_feature_contribution_grouped(importance_df, feature_names, importance_plot)

    # ── Robot simulator ────────────────────────────────────────────────────────
    if not args.no_sim:
        log_section(step, "ROBOT SIMULATOR — LIVE RISK SCORING")
        step += 1
        run_simulation(
            binary_model=best_binary_model,
            multiclass_model=best_multi_model,
            scaler=scaler,
            class_names=class_names,
            rul_model=rul_model,
            time_steps=time_steps,
            n_steps=args.sim_steps,
            output_csv=SIM_LOG_CSV,
            verbose=True,
            print_every=20,
        )

    elapsed = time.time() - total_start

    # ── PDF report ─────────────────────────────────────────────────────────────
    outputs = [
        (get_plot_path("binary_history"), "Binary LSTM training curves"),
        (get_plot_path("binary_cm"),      "Binary confusion matrix"),
        (get_plot_path("binary_roc"),     "Binary ROC curve"),
        (get_plot_path("binary_f1"),      "Binary F1 scores per class"),
        (get_plot_path("multi_history"),  "Multiclass LSTM training curves"),
        (get_plot_path("multi_cm"),       "Multiclass confusion matrix"),
        (get_plot_path("multi_roc"),      "Multiclass ROC curves (one-vs-rest)"),
        (get_plot_path("multi_f1"),       "Multiclass F1 scores per class"),
        (RESULTS_CSV,                     "Combined metrics summary"),
        (SIM_LOG_CSV,                     "Robot simulator step-by-step log"),
    ]

    save_pdf_report(
        csv_path=args.csv,
        args=args,
        input_shape=input_shape,
        class_names=class_names,
        binary_res=best_binary_res,
        multi_res=best_multi_res,
        elapsed_seconds=elapsed,
        outputs=outputs,
        out_path=REPORT_PATH,
    )

    # ── Output checklist ───────────────────────────────────────────────────────
    log_section(step, "OUTPUT FILES")
    for path, desc in outputs:
        if path == SIM_LOG_CSV and args.no_sim:
            tick = "-"
        else:
            tick = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{tick}] {path:<60} <- {desc}")

    print(f"\n  Total time: {elapsed / 60:.1f} min")
    print("\n  Done.")


if __name__ == "__main__":
    main()
