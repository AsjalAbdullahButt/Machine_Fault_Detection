"""
main.py
=======
Predictive Fault Detection in Robots Using Deep Learning
CLI orchestrator — runs all 7 pipeline steps end-to-end.
"""

import argparse
import os
import sys
import time
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from data_processing import build_pipeline
from models import (
    build_binary_lstm,
    build_multiclass_lstm,
    evaluate_binary,
    evaluate_multiclass,
    plot_training_history,
    train_model,
)
from report_utils import save_pdf_report
from simulator import run_simulation
from keras.utils import to_categorical

from config import (
    PLOTS_DIR,
    REPORT_PDF,
    SIM_LOG_CSV,
    RESULTS_CSV,
    EPOCHS,
    BATCH_SIZE,
    TIME_STEPS,
    SIM_STEPS,
)

DEFAULT_CSV = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "Dataset", "predictive_maintenance.csv")
)
REPORT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", REPORT_PDF))
os.makedirs(PLOTS_DIR, exist_ok=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predictive Fault Detection — LSTM pipeline"
    )
    parser.add_argument("--csv",       default=DEFAULT_CSV,   help="Path to dataset CSV")
    parser.add_argument("--epochs",    type=int, default=EPOCHS,      help="Max training epochs")
    parser.add_argument("--timesteps", type=int, default=TIME_STEPS,  help="LSTM sequence length")
    parser.add_argument("--batch",     type=int, default=BATCH_SIZE,  help="Batch size")
    parser.add_argument("--sim-steps", type=int, default=SIM_STEPS,   help="Simulator steps")
    parser.add_argument("--no-sim",    action="store_true",            help="Skip robot simulator")
    return parser.parse_args()


# ── Logging ───────────────────────────────────────────────────────────────────

def log_section(step: int, title: str):
    ts  = datetime.now().strftime("%H:%M:%S")
    bar = "=" * 65
    print(f"\n{bar}")
    print(f"  STEP {step} — {title}   [{ts}]")
    print(bar)


# ── Results summary ───────────────────────────────────────────────────────────

def save_results_summary(binary_res: dict, multi_res: dict, out_path: str = RESULTS_CSV):
    rows = [
        {
            "Model":     "Binary LSTM (Fault vs No Fault)",
            "Accuracy":  f"{binary_res['accuracy']:.4f}",
            "Precision": f"{binary_res['precision']:.4f}",
            "Recall":    f"{binary_res['recall']:.4f}",
            "F1":        f"{binary_res['f1']:.4f}",
            "ROC-AUC":   f"{binary_res['roc_auc']:.4f}",
        },
        {
            "Model":     "Multiclass LSTM (Failure Type)",
            "Accuracy":  f"{multi_res['accuracy']:.4f}",
            "Precision": f"{multi_res['precision']:.4f}",
            "Recall":    f"{multi_res['recall']:.4f}",
            "F1":        f"{multi_res['f1']:.4f}",
            "ROC-AUC":   "N/A (see 07_Multiclass_LSTM_ROC_Curves.png)",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    sep = "=" * 65
    print(f"\n{sep}\n  FINAL RESULTS SUMMARY\n{sep}")
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
    pipeline = build_pipeline(
        csv_path=args.csv,
        time_steps=args.timesteps,
        test_size=0.20,
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

    # ── Step 2: Binary LSTM training ──────────────────────────────────────────
    log_section(2, "BINARY LSTM — TRAINING")
    binary_model = build_binary_lstm(input_shape=input_shape)
    print(f"\n  Training on {len(X_train_bin):,} sequences ...")
    t0 = time.time()
    history_bin = train_model(
        model=binary_model, X_train=X_train_bin, y_train=y_train_bin,
        class_weights=cw_bin, epochs=args.epochs, batch_size=args.batch,
    )
    print(f"  Training time: {time.time() - t0:.1f}s")
    plot_training_history(history_bin, "Binary LSTM — Training History", "binary_history")

    # ── Step 3: Binary LSTM evaluation ───────────────────────────────────────
    log_section(3, "BINARY LSTM — EVALUATION")
    binary_results = evaluate_binary(binary_model, X_test, y_test_bin)

    # ── Step 4: Multiclass LSTM training ─────────────────────────────────────
    log_section(4, "MULTICLASS LSTM — TRAINING")
    y_train_mul_ohe = to_categorical(y_train_mul, num_classes)
    multi_model     = build_multiclass_lstm(input_shape=input_shape, num_classes=num_classes)
    print(f"\n  Training on {len(X_train_mul):,} sequences ...")
    t0 = time.time()
    history_mul = train_model(
        model=multi_model, X_train=X_train_mul, y_train=y_train_mul_ohe,
        class_weights=cw_mul, epochs=args.epochs, batch_size=args.batch,
    )
    print(f"  Training time: {time.time() - t0:.1f}s")
    plot_training_history(history_mul, "Multiclass LSTM — Training History", "multi_history")

    # ── Step 5: Multiclass LSTM evaluation ───────────────────────────────────
    log_section(5, "MULTICLASS LSTM — EVALUATION")
    multi_results = evaluate_multiclass(
        multi_model, X_test, y_test_mul, class_names, num_classes
    )

    # ── Step 6: Results summary ───────────────────────────────────────────────
    log_section(6, "RESULTS SUMMARY")
    save_results_summary(binary_results, multi_results)

    # ── Step 7: Robot simulator ───────────────────────────────────────────────
    if not args.no_sim:
        log_section(7, "ROBOT SIMULATOR — LIVE RISK SCORING")
        run_simulation(
            binary_model=binary_model,
            scaler=scaler,
            time_steps=time_steps,
            n_steps=args.sim_steps,
            output_csv=SIM_LOG_CSV,
            verbose=True,
            print_every=20,
        )

    elapsed = time.time() - total_start

    # ── Build PDF report ──────────────────────────────────────────────────────
    from config import get_plot_path
    outputs = [
        (get_plot_path("binary_history"),  "Binary LSTM training curves"),
        (get_plot_path("binary_cm"),       "Binary confusion matrix"),
        (get_plot_path("binary_roc"),      "Binary ROC curve"),
        (get_plot_path("binary_f1"),       "Binary F1 scores per class"),
        (get_plot_path("multi_history"),   "Multiclass LSTM training curves"),
        (get_plot_path("multi_cm"),        "Multiclass confusion matrix"),
        (get_plot_path("multi_roc"),       "Multiclass ROC curves (one-vs-rest)"),
        (get_plot_path("multi_f1"),        "Multiclass F1 scores per class"),
        (RESULTS_CSV,                      "Combined metrics summary"),
        (SIM_LOG_CSV,                      "Robot simulator step-by-step log"),
    ]

    save_pdf_report(
        csv_path=args.csv,
        args=args,
        input_shape=input_shape,
        class_names=class_names,
        binary_res=binary_results,
        multi_res=multi_results,
        elapsed_seconds=elapsed,
        outputs=outputs,
        out_path=REPORT_PATH,
    )

    # ── Output checklist ──────────────────────────────────────────────────────
    log_section(8, "OUTPUT FILES")
    all_outputs = outputs + [(REPORT_PATH, "Full PDF report with embedded plots")]
    for path, desc in all_outputs:
        if path.endswith(SIM_LOG_CSV) and args.no_sim:
            tick = "-"
        else:
            tick = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{tick}] {path:<55} <- {desc}")

    print(f"\n  Total time: {elapsed / 60:.1f} min")
    print("\n  Done.")


if __name__ == "__main__":
    main()
