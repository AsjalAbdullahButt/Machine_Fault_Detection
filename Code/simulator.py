"""
simulator.py — Robot Fault Simulator

Generates machine telemetry across 4 degradation phases:
  NORMAL → WARM-UP → STRESS → FAILURE-ZONE

Key parameters:
  n_steps      — total simulation steps
  phase_splits — fraction of steps per phase (must sum to 1.0)
  noise_std    — Gaussian noise std per step
  drift_scale  — multiplier on per-step drift (1.0 = nominal, >1 = faster degradation)
  random_seed  — RNG seed for reproducibility
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque

from config import (
    FAULT_THRESHOLDS,
    PHASE_SPLITS,
    NOISE_STD,
    RANDOM_SEED,
    BINARY_THRESHOLD,
    get_plot_path,
    PLOTS_DIR,
    ensure_plot_dirs,
)

ensure_plot_dirs()

# ── Warning / recommendation messages ────────────────────────────────────────
WARNING_MESSAGES = {
    "Air temperature [K]":      "HIGH AIR TEMPERATURE detected",
    "Process temperature [K]":  "HIGH PROCESS TEMPERATURE — heat dissipation risk",
    "Rotational speed [rpm]":   "LOW ROTATIONAL SPEED — possible motor overload",
    "Torque [Nm]":              "HIGH TORQUE — risk of overstrain / power failure",
    "Tool wear [min]":          "EXCESSIVE TOOL WEAR — replace tool soon",
}

RECOMMENDATIONS = {
    "GREEN":  "System operating normally. Continue current task.",
    "YELLOW": "Elevated risk. Reduce load and increase monitoring frequency.",
    "RED":    "CRITICAL — Stop robot immediately and perform maintenance.",
}

TIER_COLORS = {
    "GREEN":  "\033[92m",
    "YELLOW": "\033[93m",
    "RED":    "\033[91m",
}
RESET = "\033[0m"


# ── Simulator ─────────────────────────────────────────────────────────────────

class RobotSimulator:
    """Simulates machine degradation across four sequential phases."""

    BASE = {
        "Type":                       1,
        "Air temperature [K]":      300.0,
        "Process temperature [K]":  310.0,
        "Rotational speed [rpm]":  1538.0,
        "Torque [Nm]":               39.6,
        "Tool wear [min]":            0.0,
    }

    DRIFT = {
        "Air temperature [K]":      +0.04,
        "Process temperature [K]":  +0.03,
        "Rotational speed [rpm]":   -1.2,
        "Torque [Nm]":              +0.18,
        "Tool wear [min]":          +1.5,
    }

    def __init__(
        self,
        n_steps: int        = 200,
        phase_splits: tuple = PHASE_SPLITS,
        noise_std: float    = NOISE_STD,
        drift_scale: float  = 1.0,
        random_seed: int    = RANDOM_SEED,
    ):
        if abs(sum(phase_splits) - 1.0) > 1e-5:
            raise ValueError(f"phase_splits must sum to 1.0, got {sum(phase_splits):.4f}")
        if drift_scale <= 0:
            raise ValueError(f"drift_scale must be positive, got {drift_scale}")

        self.n_steps    = n_steps
        self.noise_std  = noise_std
        self.drift_scale = float(drift_scale)
        np.random.seed(random_seed)

        cuts = np.cumsum([int(f * n_steps) for f in phase_splits])
        self.phase_ends = {
            "NORMAL":       int(cuts[0]),
            "WARM-UP":      int(cuts[1]),
            "STRESS":       int(cuts[2]),
            "FAILURE-ZONE": n_steps,
        }
        self._state = {k: v for k, v in self.BASE.items()}
        self._step  = 0

    def _phase(self) -> str:
        s = self._step
        if s < self.phase_ends["NORMAL"]:    return "NORMAL"
        elif s < self.phase_ends["WARM-UP"]: return "WARM-UP"
        elif s < self.phase_ends["STRESS"]:  return "STRESS"
        else:                                return "FAILURE-ZONE"

    def _apply_drift(self, phase: str) -> None:
        phase_multipliers = {
            "NORMAL":       0.0,
            "WARM-UP":      0.3,
            "STRESS":       1.0,
            "FAILURE-ZONE": 2.5,
        }
        m = phase_multipliers.get(phase, 0.0) * self.drift_scale
        if m > 0:
            for k in self.DRIFT:
                self._state[k] += self.DRIFT[k] * m

    def next_step(self) -> dict:
        """Advance one step and return the observation dict."""
        phase = self._phase()
        self._apply_drift(phase)

        obs = {}
        for feat, val in self._state.items():
            if feat == "Type":
                obs[feat] = int(val)
            else:
                obs[feat] = float(max(0.0, val + np.random.normal(0, self.noise_std)))

        self._step += 1
        obs["step"]  = self._step
        obs["phase"] = phase
        return obs

    def step(self) -> dict:
        """Alias for next_step() — adds normalised RUL and degradation fields."""
        obs = self.next_step()
        steps_left       = max(0, self.n_steps - self._step)
        obs["rul"]         = float(steps_left)
        obs["degradation"] = float(self._step / self.n_steps)
        return obs

    def reset(self) -> None:
        self._state = {k: v for k, v in self.BASE.items()}
        self._step  = 0

    def __iter__(self):
        for _ in range(self.n_steps):
            yield self.next_step()

    def __len__(self) -> int:
        return self.n_steps


# ── Risk scoring ──────────────────────────────────────────────────────────────

def score_risk(risk_prob: float, obs: dict) -> dict:
    """Compute tiered risk level and per-feature threshold warnings."""
    pct  = float(np.clip(risk_prob * 100.0, 0, 100))
    tier = "GREEN" if pct < 30 else "YELLOW" if pct < 70 else "RED"

    warnings = []
    for feat, threshold in FAULT_THRESHOLDS.items():
        val = obs.get(feat)
        if val is None:
            continue
        if feat == "Rotational speed [rpm]":
            if float(val) < threshold:
                warnings.append(WARNING_MESSAGES[feat])
        else:
            if float(val) > threshold:
                warnings.append(WARNING_MESSAGES[feat])

    return {
        "risk_pct":       round(pct, 2),
        "tier":           tier,
        "warnings":       warnings,
        "recommendation": RECOMMENDATIONS[tier],
    }


# ── Terminal reporter ─────────────────────────────────────────────────────────

def print_risk_report(step, phase, obs, risk, risk_delta, predicted_failure_type, predicted_rul, verbose=True):
    if not verbose:
        return
    tier  = risk["tier"]
    color = TIER_COLORS.get(tier, "")
    print(f"\n{'─'*60}")
    print(f"  Step {step:>3} | Phase: {phase:<12} | "
          f"{color}RISK: {risk['risk_pct']:>5.1f}%  [{tier}]{RESET}")
    print(f"  ΔRisk(5-step): {risk_delta:+6.2f}%   Pred Failure: {predicted_failure_type}")
    if predicted_rul is not None:
        print(f"  Estimated RUL: {predicted_rul:>6.1f} steps")
    print(f"  Air Temp : {obs.get('Air temperature [K]', 0):>7.2f} K   "
          f"Proc Temp: {obs.get('Process temperature [K]', 0):>7.2f} K")
    print(f"  RPM      : {obs.get('Rotational speed [rpm]', 0):>7.1f}     "
          f"Torque   : {obs.get('Torque [Nm]', 0):>7.2f} Nm")
    print(f"  Tool Wear: {obs.get('Tool wear [min]', 0):>7.1f} min")
    for w in risk.get("warnings", []):
        print(f"  {color}[!] {w}{RESET}")
    print(f"  -> {risk['recommendation']}")


# ── Risk timeline plot ────────────────────────────────────────────────────────

def plot_risk_timeline(df_log: pd.DataFrame):
    """Save risk timeline with rolling risk-delta acceleration subplot."""
    valid = df_log[df_log["tier"].isin(["GREEN", "YELLOW", "RED"])].copy()
    if valid.empty:
        print("  [WARN] plot_risk_timeline: no scored rows to plot.")
        return

    color_map = {"GREEN": "#2ecc71", "YELLOW": "#f39c12", "RED": "#e74c3c"}
    path      = get_plot_path("sim_timeline")

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 7), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    ax = axes[0]

    for _, row in valid.iterrows():
        ax.bar(row["step"], row["risk_pct"],
               color=color_map.get(row["tier"], "#aaa"),
               width=1.0, linewidth=0)

    phase_changes = valid[valid["phase"] != valid["phase"].shift()]
    for _, row in phase_changes.iterrows():
        ax.axvline(row["step"], color="#555", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.text(row["step"] + 0.5, 95, row["phase"], fontsize=7, color="#333", va="top")

    ax.axhline(30, color="#f39c12", linewidth=1, linestyle=":", alpha=0.7, label="Yellow threshold (30%)")
    ax.axhline(70, color="#e74c3c", linewidth=1, linestyle=":", alpha=0.7, label="Red threshold (70%)")
    ax.set_xlim(valid["step"].min(), valid["step"].max())
    ax.set_ylim(0, 105)
    ax.set_ylabel("Fault Risk (%)")
    ax.set_title("Robot Simulator — Fault Risk Timeline", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax2 = axes[1]
    deltas      = valid["risk_delta"].fillna(0.0).values if "risk_delta" in valid.columns else np.zeros(len(valid))
    steps       = valid["step"].values
    accel_colors = [
        "#e74c3c" if d >= 10 else "#f39c12" if d >= 3 else "#2ecc71"
        for d in deltas
    ]
    ax2.bar(steps, deltas, color=accel_colors, width=1.0, linewidth=0)
    ax2.axhline(0,  color="#444", linewidth=1)
    ax2.axhline(3,  color="#f39c12", linewidth=1, linestyle=":", alpha=0.8)
    ax2.axhline(10, color="#e74c3c", linewidth=1, linestyle=":", alpha=0.8)
    ax2.set_ylabel("Δ Risk (%)")
    ax2.set_xlabel("Simulation Step")
    ax2.set_title("Rolling Risk Delta: risk[t] − mean(risk[t−5:t])", fontsize=10)
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [Saved] {path}")


# ── Main run function ─────────────────────────────────────────────────────────

def run_simulation(
    binary_model,
    multiclass_model,
    scaler,
    class_names,
    rul_model=None,
    binary_threshold: float = BINARY_THRESHOLD,
    time_steps: int  = 10,
    n_steps: int     = 200,
    output_csv: str  = "simulation_log.csv",
    verbose: bool    = True,
    print_every: int = 10,
    drift_scale: float = 1.0,
    noise_std: float = NOISE_STD,
):
    """Run the robot simulator and score each step with the trained binary LSTM."""
    print("\n" + "=" * 60)
    print("   ROBOT SIMULATOR — LIVE FAULT RISK SCORING")
    print("=" * 60)
    print(f"  Phases      : NORMAL -> WARM-UP -> STRESS -> FAILURE-ZONE")
    print(f"  Steps       : {n_steps}")
    print(f"  Window      : {time_steps} steps")
    print(f"  Drift scale : {drift_scale:.2f}x")
    print(f"  Log         : {output_csv}\n")

    sim = RobotSimulator(n_steps=n_steps, drift_scale=drift_scale, noise_std=noise_std)
    feat_cols = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    window      = deque(maxlen=time_steps)
    risk_window = deque(maxlen=5)
    log_rows    = []

    for obs in sim:
        step  = obs["step"]
        phase = obs["phase"]

        raw_vec    = np.array([[obs.get(f, 0) for f in feat_cols]], dtype=np.float32)
        scaled_vec = scaler.transform(raw_vec)[0]
        window.append(scaled_vec)

        if len(window) < time_steps:
            log_rows.append({
                "step": step, "phase": phase,
                **{f: obs.get(f, 0) for f in feat_cols},
                "risk_pct": None, "tier": "WARMING_UP",
                "warnings": "", "risk_delta": None,
                "predicted_failure_type": "WARMING_UP",
                "predicted_rul_steps": None,
                "recommendation": "Collecting initial window ...",
            })
            continue

        seq       = np.array(list(window), dtype=np.float32)[np.newaxis, ...]
        risk_prob = float(binary_model.predict(seq, verbose=0)[0][0])
        risk      = score_risk(risk_prob, obs)

        risk_window.append(risk["risk_pct"])
        rolling_mean = float(np.mean(list(risk_window)[:-1])) if len(risk_window) > 1 else risk["risk_pct"]
        risk_delta   = float(risk["risk_pct"] - rolling_mean)

        predicted_failure_type = "No Failure"
        if risk_prob >= binary_threshold:
            multi_prob = multiclass_model.predict(seq, verbose=0)[0]
            pred_class = int(np.argmax(multi_prob))
            # If top prediction is No-Failure (class 0), fall back to next-best
            if pred_class == 0 and len(multi_prob) > 1:
                pred_class = int(np.argmax(multi_prob[1:]) + 1)
            if pred_class < len(class_names):
                predicted_failure_type = class_names[pred_class]

        predicted_rul = None
        if rul_model is not None:
            raw_pred  = rul_model.predict(seq, verbose=0)
            predicted_rul = float(max(0.0, float(raw_pred[0][0])))

        should_print = verbose and (step % print_every == 0 or risk["tier"] == "RED")
        print_risk_report(step, phase, obs, risk, risk_delta,
                          predicted_failure_type, predicted_rul, verbose=should_print)

        log_rows.append({
            "step":  step, "phase": phase,
            **{f: round(float(obs.get(f, 0)), 3) for f in feat_cols},
            "risk_pct":               risk["risk_pct"],
            "risk_delta":             round(risk_delta, 3),
            "tier":                   risk["tier"],
            "predicted_failure_type": predicted_failure_type,
            "predicted_rul_steps":    round(predicted_rul, 3) if predicted_rul is not None else None,
            "warnings":               " | ".join(risk["warnings"]),
            "recommendation":         risk["recommendation"],
        })

    df_log = pd.DataFrame(log_rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_log.to_csv(output_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  Simulation complete. Log saved -> {output_csv}")

    valid = df_log[df_log["tier"].isin(["GREEN", "YELLOW", "RED"])]
    if len(valid):
        tc = valid["tier"].value_counts()
        print(f"\n  Risk tier breakdown:")
        for t in ["GREEN", "YELLOW", "RED"]:
            cnt = tc.get(t, 0)
            bar = chr(9608) * int(cnt / max(len(valid), 1) * 30)
            print(f"    {t:<8}: {cnt:>4} steps  {bar}")

    plot_risk_timeline(df_log)
    print("=" * 60)
    return df_log
