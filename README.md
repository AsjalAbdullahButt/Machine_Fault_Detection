# Predictive Fault Detection in Robots Using Deep Learning

End-to-end LSTM pipeline for predictive maintenance — detects machine faults and classifies failure types from industrial sensor telemetry.

## Developed by Asjal Abdullah

---

## Overview

This project builds a two-stage deep learning system trained on the [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020):

1. **Binary LSTM** — classifies each timestep as *Fault* or *No Fault*
2. **Multiclass LSTM** — identifies the specific failure type among six categories

A four-phase robot simulator then applies the trained binary model in real time, scoring live risk across a degradation scenario (NORMAL → WARM-UP → STRESS → FAILURE-ZONE).

---

## Project Structure

```text
Machine_Fault_Detection/
├── Code/
│   ├── config.py            # all hyperparameters and path constants
│   ├── data_processing.py   # loading, SMOTE, scaling, sequence creation
│   ├── models.py            # LSTM architectures, training, evaluation, plots
│   ├── simulator.py         # 4-phase robot simulator with live risk scoring
│   ├── report_utils.py      # PDF report builder (metrics + embedded plots)
│   └── main.py              # CLI orchestrator
├── Dataset/
│   └── predictive_maintenance.csv
├── plots/                   # generated after a run
├── Report.pdf               # generated after a run
├── results_summary.csv      # generated after a run
├── requirements.txt
└── README.md
```

---

## Requirements

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn tensorflow keras reportlab
```

---

## Usage

### Full pipeline (60 epochs, 200 simulator steps)

```bash
python Code/main.py
```

### Quick smoke test (fast, no simulator)

```bash
python Code/main.py --epochs 5 --no-sim
```

### Custom run

```bash
python Code/main.py --epochs 60 --timesteps 10 --batch 64 --sim-steps 200
```

### Custom dataset path

```bash
python Code/main.py --csv /path/to/predictive_maintenance.csv
```

---

## Configuration

All hyperparameters live in `Code/config.py`. Key values:

| Parameter | Default | Description |
| --- | --- | --- |
| `LSTM_UNITS` | 64 | Units in first LSTM layer |
| `DROPOUT_RATE` | 0.3 | Dropout applied after each LSTM block |
| `BINARY_THRESHOLD` | 0.4 | Decision boundary for fault detection |
| `TIME_STEPS` | 10 | Sliding window length |
| `EPOCHS` | 60 | Maximum training epochs (early stopping) |
| `BATCH_SIZE` | 64 | Training batch size |

---

## Output Files

After a successful run:

| File | Description |
| --- | --- |
| `Report.pdf` | Full report with metrics tables and all 8 embedded plots |
| `results_summary.csv` | Binary and multiclass metric comparison |
| `plots/01_Binary_LSTM_Training_History.png` | Loss & accuracy curves - binary model |
| `plots/02_Binary_LSTM_Confusion_Matrix.png` | Confusion matrix - binary model |
| `plots/03_Binary_LSTM_ROC_Curve.png` | ROC curve with AUC - binary model |
| `plots/04_Binary_LSTM_F1_Scores.png` | Per-class F1 bar chart - binary model |
| `plots/05_Multiclass_LSTM_Training_History.png` | Loss & accuracy curves - multiclass model |
| `plots/06_Multiclass_LSTM_Confusion_Matrix.png` | Confusion matrix - multiclass model |
| `plots/07_Multiclass_LSTM_ROC_Curves.png` | One-vs-rest ROC curves - multiclass model |
| `plots/08_Multiclass_LSTM_F1_Scores.png` | Per-class F1 bar chart - multiclass model |
| `plots/09_Simulation_Risk_Timeline.png` | Color-coded risk % over simulator steps |
| `simulation_log.csv` | Step-by-step simulator log with risk tier and warnings |

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** — 10,000 records, 6 features:

| Feature | Description |
| --- | --- |
| Air temperature [K] | Ambient air temperature |
| Process temperature [K] | Operating process temperature |
| Rotational speed [rpm] | Spindle rotation speed |
| Torque [Nm] | Applied torque |
| Tool wear [min] | Cumulative tool wear time |
| Type | Machine grade (L / M / H) |

Target: binary fault label + one of six failure types (No Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, Tool Wear Failure, Random Failures).

Class imbalance (~97% no-fault) is addressed with **SMOTE** combined with **class weights**.

---

## Notes

- The dataset must be at `Dataset/predictive_maintenance.csv` or passed via `--csv`.
- All plots and generated files are excluded from git by default (see `.gitignore`).
- To tune the model, edit `Code/config.py` — no need to touch any other file.
