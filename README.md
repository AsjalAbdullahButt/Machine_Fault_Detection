# Predictive Fault Detection in Robots Using Deep Learning 🤖

End-to-end deep learning project for predictive maintenance and failure analysis on industrial sensor telemetry.

## Overview

This project goes beyond a single model. It compares multiple architectures, explains their predictions, and simulates live risk over time:

1. Binary fault detection: `Fault` vs `No Fault`
2. Multiclass failure classification: one of six failure types
3. RUL estimation: regression head that predicts remaining useful life
4. Explainability: attention heatmaps and feature contribution charts
5. Simulation: live risk scoring, failure-type prediction, and rolling risk acceleration
6. Model comparison: a summary table across LSTM, CNN+LSTM, and Transformer

## Pipeline Summary

The project uses this workflow:

1. Load and clean the dataset from `Dataset/`
2. Encode features and labels
3. Build temporal sequences for the LSTM-style models
4. Apply sequence-level SMOTE only after sequencing to avoid temporal leakage
5. Use `TimeSeriesSplit` for fold-aware validation
6. Train and compare three architectures:
   - LSTM
   - CNN + LSTM
   - Lightweight Transformer
7. Train an additional RUL regression model with Huber loss
8. Generate evaluation plots, explainability plots, and a PDF report
9. Run the robot simulator with binary risk, failure type, and RUL outputs

## Models

### 1) LSTM

Baseline recurrent model for sequential fault detection and classification. It is the reference point for all comparisons.

### 2) CNN + LSTM

This hybrid model uses `Conv1D` to learn short local sensor patterns, then `LSTM` to capture longer temporal dependencies. In practice, this is often stronger than a pure LSTM on sensor streams.

### 3) Lightweight Transformer

This model uses a small self-attention encoder with two attention heads. Instead of relying on recurrence, it learns global relationships across timesteps. It is also the model used for attention-based explainability.

### 4) RUL Regression LSTM

This model predicts remaining useful life in steps rather than only yes/no fault status. It uses Huber loss for robustness and is shown in the simulator alongside fault risk.

## Explainability

The project generates two explainability outputs:

- Attention heatmaps for the Transformer, one per failure type
- Permutation importance plots showing which features matter most for each failure class

These are saved under `plots/` and also included in the report workflow.

## Project Structure 📁

```text
.
├── Code/
│   ├── advanced_models.py     # RUL regression model and evaluation
│   ├── config.py              # hyperparameters, paths, thresholds
│   ├── data_processing.py     # preprocessing, SMOTE, sequence building, CV folds
│   ├── explainability.py      # attention heatmaps and feature contribution plots
│   ├── main.py                # CLI entrypoint and orchestration
│   ├── models.py              # LSTM / CNN+LSTM / Transformer models and metrics
│   ├── report_utils.py        # PDF report generator
│   ├── simulator.py           # live risk scoring and failure-type reporting
│   └── tuning.py              # manual hyperparameter grid search
├── Dataset/
│   └── predictive_maintenance.csv
├── Documentation/
├── models/
├── plots/
├── Report.pdf
├── results_summary.csv
├── requirements.txt
└── README.md
```

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

If you want a manual install, use:

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn tensorflow keras reportlab
```

## Run Commands

All commands below are generic and do not depend on a local machine path.

### Full run

```bash
python Code/main.py
```

### Faster smoke run

```bash
python Code/main.py --epochs 1 --no-sim --no-tune
```

### Skip retraining and load saved models

```bash
python Code/main.py --load-models
```

### Disable simulator

```bash
python Code/main.py --no-sim
```

### Custom dataset path

```bash
python Code/main.py --csv /path/to/predictive_maintenance.csv
```

### Skip tuning

```bash
python Code/main.py --no-tune
```

## Hyperparameters

Key values are centralized in `Code/config.py`.

| Parameter | Meaning |
| --- | --- |
| `TIME_STEPS` | Sliding window size for sequences |
| `EPOCHS` | Maximum training epochs |
| `BATCH_SIZE` | Training batch size |
| `ES_PATIENCE` | Early stopping patience |
| `ES_MIN_DELTA` | Minimum validation loss improvement |
| `BINARY_THRESHOLD` | Decision threshold for fault detection |
| `TS_CV_SPLITS` | Number of time-series CV folds |

The manual tuning grid searches over:

- `LSTM_UNITS` in `{32, 64, 128}`
- `DROPOUT_RATE` in `{0.2, 0.3, 0.4}`
- `LEARNING_RATE` in `{1e-3, 5e-4}`

## Data and Evaluation Process

### Data handling

- SMOTE is applied after sequence creation to preserve temporal structure
- `TimeSeriesSplit` is used so future data does not leak into training folds
- Class weights are still computed and used during training

### Evaluation

- Binary metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, calibration, and Brier score
- Multiclass metrics: Accuracy, weighted Precision/Recall/F1, ROC-AUC, PR-AUC, and MCC
- Model comparison table: LSTM vs CNN+LSTM vs Transformer
- RUL metrics: MAE and RMSE

## Outputs 📁

After a run, the following artifacts are produced:

| File | Description |
| --- | --- |
| `Report.pdf` | Full PDF report with metrics and plots |
| `results_summary.csv` | Comparison table across all architectures |
| `simulation_log.csv` | Live simulator log with risk, delta, failure type, and RUL |
| `models/` | Saved trained models |
| `plots/` | Training, evaluation, explainability, and simulator plots |

## Dataset

The project uses the [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020).

Features include:

- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Type

Targets include:

- Binary fault label
- Failure type label

## Notes

- Put the dataset in `Dataset/` or pass it with `--csv`.
- Saved models are written to `models/`.
- Generated plots and reports are written to `plots/`, `Report.pdf`, and `results_summary.csv`.
- The simulator now reports predicted failure type and RUL when available.
