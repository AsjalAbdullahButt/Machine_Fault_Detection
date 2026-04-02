"""
main.py
=======
Predictive Fault Detection in Robots Using Deep Learning
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
    PLOTS_DIR,
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

DEFAULT_CSV = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "Dataset", "predictive_maintenance.csv")
)
REPORT_PDF = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "Report.pdf"))


def parse_args():
    parser = argparse.ArgumentParser(description="Predictive Fault Detection - LSTM pipeline")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to predictive_maintenance.csv")
    parser.add_argument("--epochs", type=int, default=60, help="Max training epochs")
    parser.add_argument("--timesteps", type=int, default=10, help="LSTM sequence length")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--sim-steps", type=int, default=200, help="Robot simulator steps")
    parser.add_argument("--no-sim", action="store_true", help="Skip robot simulator")
    return parser.parse_args()


def print_section(title: str):
    bar = "═" * 65
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def save_results_summary(binary_res: dict, multi_res: dict, out_path: str = "results_summary.csv"):
    rows = [
        {
            "Model": "Binary LSTM (Fault vs No Fault)",
            "Accuracy": f"{binary_res['accuracy']:.4f}",
            "Precision": f"{binary_res['precision']:.4f}",
            "Recall": f"{binary_res['recall']:.4f}",
            "F1": f"{binary_res['f1']:.4f}",
            "ROC-AUC": f"{binary_res['roc_auc']:.4f}",
        },
        {
            "Model": "Multiclass LSTM (Failure Type)",
            "Accuracy": f"{multi_res['accuracy']:.4f}",
            "Precision": f"{multi_res['precision']:.4f}",
            "Recall": f"{multi_res['recall']:.4f}",
            "F1": f"{multi_res['f1']:.4f}",
            "ROC-AUC": "N/A (see multiclass_roc.png)",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    sep = "═" * 65
    print(f"\n{sep}")
    print("  FINAL RESULTS SUMMARY")
    print(sep)
    print(df.to_string(index=False))
    print(f"\n  Saved → {out_path}")
    print(sep)


def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] Dataset not found at: {args.csv}")
        print("  Place predictive_maintenance.csv in the same folder as main.py,")
        print("  or pass --csv /path/to/file.csv")
        sys.exit(1)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    total_start = time.time()

    print_section("STEP 1 — DATA PIPELINE")
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
    X_test = pipeline["X_test"]
    y_test_bin = pipeline["y_test_bin"]
    y_test_mul = pipeline["y_test_mul"]
    scaler = pipeline["scaler"]
    le = pipeline["label_encoder"]
    cw_bin = pipeline["class_weights_binary"]
    cw_mul = pipeline["class_weights_multi"]
    n_features = pipeline["n_features"]
    num_classes = pipeline["num_classes"]
    time_steps = pipeline["time_steps"]

    class_names = list(le.classes_)
    input_shape = (time_steps, n_features)
    print(f"\n  Input shape  : {input_shape}")
    print(f"  Num classes  : {num_classes}  → {class_names}")

    print_section("STEP 2 — BINARY LSTM — TRAINING")
    binary_model = build_binary_lstm(input_shape=input_shape, units=64, dropout=0.3)
    print(f"\n  Training on {len(X_train_bin):,} sequences …")
    t0 = time.time()
    history_bin = train_model(
        model=binary_model,
        X_train=X_train_bin,
        y_train=y_train_bin,
        class_weights=cw_bin,
        epochs=args.epochs,
        batch_size=args.batch,
        val_split=0.1,
    )
    print(f"  Training time: {time.time() - t0:.1f}s")
    plot_training_history(
        history_bin,
        title="Binary LSTM — Training History",
        save_name="binary_training_history.png",
    )

    print_section("STEP 3 — BINARY LSTM — EVALUATION")
    binary_results = evaluate_binary(binary_model, X_test, y_test_bin, threshold=0.4)

    print_section("STEP 4 — MULTICLASS LSTM — TRAINING")
    y_train_mul_ohe = to_categorical(y_train_mul, num_classes)
    multi_model = build_multiclass_lstm(
        input_shape=input_shape,
        num_classes=num_classes,
        units=64,
        dropout=0.3,
    )
    print(f"\n  Training on {len(X_train_mul):,} sequences …")
    t0 = time.time()
    history_mul = train_model(
        model=multi_model,
        X_train=X_train_mul,
        y_train=y_train_mul_ohe,
        class_weights=cw_mul,
        epochs=args.epochs,
        batch_size=args.batch,
        val_split=0.1,
    )
    print(f"  Training time: {time.time() - t0:.1f}s")
    plot_training_history(
        history_mul,
        title="Multiclass LSTM — Training History",
        save_name="multiclass_training_history.png",
    )

    print_section("STEP 5 — MULTICLASS LSTM — EVALUATION")
    multi_results = evaluate_multiclass(
        multi_model,
        X_test,
        y_test_mul,
        class_names,
        num_classes,
    )

    print_section("STEP 6 — RESULTS SUMMARY")
    save_results_summary(binary_results, multi_results, "results_summary.csv")

    if not args.no_sim:
        print_section("STEP 7 — ROBOT SIMULATOR — LIVE RISK SCORING")
        run_simulation(
            binary_model=binary_model,
            scaler=scaler,
            time_steps=time_steps,
            n_steps=args.sim_steps,
            output_csv="simulation_log.csv",
            verbose=True,
            print_every=20,
        )

    elapsed = time.time() - total_start
    print(f"\n  Total elapsed time: {elapsed / 60:.1f} min")

    outputs = [
        ("plots/binary_training_history.png", "Binary LSTM loss & accuracy curves"),
        ("plots/binary_cm.png", "Binary confusion matrix"),
        ("plots/binary_roc.png", "Binary ROC curve"),
        ("plots/binary_f1.png", "Binary F1 bar chart"),
        ("plots/multiclass_training_history.png", "Multiclass training curves"),
        ("plots/multiclass_cm.png", "Multiclass confusion matrix"),
        ("plots/multiclass_roc.png", "Multiclass ROC curves (one-vs-rest)"),
        ("plots/multiclass_f1.png", "Multiclass F1 bar chart"),
        ("results_summary.csv", "Combined metrics table"),
        ("simulation_log.csv", "Robot simulator step-by-step log"),
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
        out_path=REPORT_PDF,
    )

    print_section("OUTPUT FILES")
    outputs_with_report = outputs + [(REPORT_PDF, "Polished PDF report")]
    for path, desc in outputs_with_report:
        if path.endswith("simulation_log.csv") and args.no_sim:
            tick = "-"
        else:
            tick = "✓" if os.path.exists(path) else "✗"
        print(f"  [{tick}] {path:<45}  ← {desc}")

    print("\n  Done. ✅")


if __name__ == "__main__":
    main()
