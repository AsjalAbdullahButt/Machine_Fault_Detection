"""
simulator.py
============
Python-based robot / machine simulator.

Strategy chosen: **Hybrid scenario-driven with gradual parameter drift**.

The simulator runs N steps across 4 operating phases that realistically
mirror real-world machine degradation:

  Phase 1 — NORMAL      : stable parameters within safe operating range
  Phase 2 — WARM-UP     : slight temp / torque rise as load increases
  Phase 3 — STRESS      : progressive drift toward failure-boundary values
  Phase 4 — FAILURE-ZONE: parameters cross known fault thresholds

At each step the trained LSTM model is called, and a tiered risk report is
printed to the terminal AND saved to a CSV log file.

Risk tiers
----------
  GREEN  (0 – 30%)  : Normal operation
  YELLOW (30 – 70%) : Elevated risk — monitor closely
  RED    (70 – 100%): Critical — take immediate action

Per-feature warnings are triggered when any feature crosses its
fault-threshold (derived from dataset statistics).
"""

import os
import time
import numpy as np
import pandas as pd
from collections import deque

# ── Feature / threshold constants ────────────────────────────────────────────
FEATURE_NAMES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Approximate fault-boundary values derived from dataset EDA
FAULT_THRESHOLDS = {
    "Air temperature [K]":      301.5,   # 75th percentile in fault cases
    "Process temperature [K]":  310.5,
    "Rotational speed [rpm]":   1420,    # low speed → overload risk
    "Torque [Nm]":               48.0,   # fault mean = 50.17
    "Tool wear [min]":          180,     # fault mean = 143, max safe ≈ 200
}

WARNING_MESSAGES = {
    "Air temperature [K]":      "⚠  HIGH AIR TEMPERATURE detected",
    "Process temperature [K]":  "⚠  HIGH PROCESS TEMPERATURE — heat dissipation risk",
    "Rotational speed [rpm]":   "⚠  LOW ROTATIONAL SPEED — possible motor overload",
    "Torque [Nm]":              "⚠  HIGH TORQUE — risk of overstrain / power failure",
    "Tool wear [min]":          "⚠  EXCESSIVE TOOL WEAR — replace tool soon",
}

RECOMMENDATIONS = {
    "GREEN":  "✅ System operating normally. Continue current task.",
    "YELLOW": "⚡ Elevated risk. Reduce load, increase monitoring frequency.",
    "RED":    "🔴 CRITICAL — Stop robot immediately & perform maintenance.",
}


# ── Parameter profiles ────────────────────────────────────────────────────────

class RobotSimulator:
    """
    Generates machine telemetry in 4 sequential phases.
    Each step yields one feature vector that mimics real sensor data.
    """

    # Normal operating range (mean ± std from dataset)
    BASE = {
        "Type":                       1,      # Medium-grade machine
        "Air temperature [K]":      300.0,
        "Process temperature [K]":  310.0,
        "Rotational speed [rpm]":  1538.0,
        "Torque [Nm]":               39.6,
        "Tool wear [min]":            0.0,
    }

    # Per-step drift rates (applied during STRESS phase)
    DRIFT = {
        "Air temperature [K]":      +0.04,
        "Process temperature [K]":  +0.03,
        "Rotational speed [rpm]":   -1.2,
        "Torque [Nm]":              +0.18,
        "Tool wear [min]":          +1.5,
    }

    def __init__(
        self,
        n_steps: int = 200,
        phase_splits: tuple = (0.25, 0.20, 0.35, 0.20),
        noise_std: float = 0.3,
        random_seed: int = 42,
    ):
        assert abs(sum(phase_splits) - 1.0) < 1e-6
        self.n_steps = n_steps
        self.noise_std = noise_std
        np.random.seed(random_seed)

        cuts = np.cumsum([int(f * n_steps) for f in phase_splits])
        self.phase_ends = {
            "NORMAL":      cuts[0],
            "WARM-UP":     cuts[1],
            "STRESS":      cuts[2],
            "FAILURE-ZONE": n_steps,
        }

        self._state = {k: v for k, v in self.BASE.items()}
        self._step = 0

    def _phase(self):
        s = self._step
        if s < self.phase_ends["NORMAL"]:
            return "NORMAL"
        elif s < self.phase_ends["WARM-UP"]:
            return "WARM-UP"
        elif s < self.phase_ends["STRESS"]:
            return "STRESS"
        else:
            return "FAILURE-ZONE"

    def next_step(self) -> dict:
        phase = self._phase()

        # Apply drift per phase
        if phase == "WARM-UP":
            for k in self.DRIFT:
                self._state[k] += self.DRIFT[k] * 0.3

        elif phase == "STRESS":
            for k in self.DRIFT:
                self._state[k] += self.DRIFT[k]

        elif phase == "FAILURE-ZONE":
            for k in self.DRIFT:
                self._state[k] += self.DRIFT[k] * 2.5   # accelerated degradation

        # Add Gaussian noise to continuous features
        obs = {}
        for feat, val in self._state.items():
            if feat == "Type":
                obs[feat] = int(val)
            else:
                noise = np.random.normal(0, self.noise_std)
                obs[feat] = max(0.0, val + noise)

        self._step += 1
        obs["step"] = self._step
        obs["phase"] = phase
        return obs

    def __iter__(self):
        for _ in range(self.n_steps):
            yield self.next_step()


# ── Risk scoring ──────────────────────────────────────────────────────────────

def score_risk(risk_prob: float, obs: dict) -> dict:
    """Compute tiered risk level + per-feature warnings."""
    pct = risk_prob * 100.0

    if pct < 30:
        tier = "GREEN"
    elif pct < 70:
        tier = "YELLOW"
    else:
        tier = "RED"

    warnings = []
    for feat, threshold in FAULT_THRESHOLDS.items():
        val = obs.get(feat, 0)
        if feat == "Rotational speed [rpm]":
            if val < threshold:
                warnings.append(WARNING_MESSAGES[feat])
        else:
            if val > threshold:
                warnings.append(WARNING_MESSAGES[feat])

    recommendation = RECOMMENDATIONS[tier]

    return {
        "risk_pct": round(pct, 2),
        "tier": tier,
        "warnings": warnings,
        "recommendation": recommendation,
    }


# ── Pretty terminal printer ───────────────────────────────────────────────────

TIER_COLORS = {
    "GREEN":  "\033[92m",   # bright green
    "YELLOW": "\033[93m",   # bright yellow
    "RED":    "\033[91m",   # bright red
}
RESET = "\033[0m"


def print_risk_report(step: int, phase: str, obs: dict, risk: dict, verbose: bool = True):
    if not verbose:
        return
    tier = risk["tier"]
    color = TIER_COLORS.get(tier, "")
    print(f"\n{'─'*60}")
    print(f"  Step {step:>3} | Phase: {phase:<12} | "
          f"{color}RISK: {risk['risk_pct']:>5.1f}%  [{tier}]{RESET}")
    print(f"  Air Temp : {obs['Air temperature [K]']:>7.2f} K   "
          f"Proc Temp: {obs['Process temperature [K]']:>7.2f} K")
    print(f"  RPM      : {obs['Rotational speed [rpm]']:>7.1f}     "
          f"Torque   : {obs['Torque [Nm]']:>7.2f} Nm")
    print(f"  Tool Wear: {obs['Tool wear [min]']:>7.1f} min")
    for w in risk["warnings"]:
        print(f"  {color}{w}{RESET}")
    print(f"  → {risk['recommendation']}")


# ── Main run function ─────────────────────────────────────────────────────────

def run_simulation(
    binary_model,
    scaler,
    time_steps: int = 10,
    n_steps: int = 200,
    output_csv: str = "simulation_log.csv",
    verbose: bool = True,
    print_every: int = 10,
):
    """
    Run the robot simulator and apply the trained binary LSTM at each step.

    Parameters
    ----------
    binary_model   : trained Keras binary LSTM model
    scaler         : fitted StandardScaler from training
    time_steps     : sequence window length (must match training)
    n_steps        : total simulation steps
    output_csv     : path to save the step-by-step risk log
    verbose        : whether to print live reports
    print_every    : print every N steps (reduces terminal spam)
    """

    print("\n" + "═" * 60)
    print("   ROBOT SIMULATOR — LIVE FAULT RISK SCORING")
    print("═" * 60)
    print(f"  Phases: NORMAL → WARM-UP → STRESS → FAILURE-ZONE")
    print(f"  Total steps : {n_steps}")
    print(f"  Time window : {time_steps} steps")
    print(f"  Output log  : {output_csv}\n")

    sim = RobotSimulator(n_steps=n_steps)

    # Feature columns (same order as scaler was fit on)
    feat_cols = [
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    window = deque(maxlen=time_steps)   # sliding window of scaled feature vectors
    log_rows = []

    for obs in sim:
        step = obs["step"]
        phase = obs["phase"]

        # Extract feature vector (same order as training)
        raw_vec = np.array([[obs[f] for f in feat_cols]], dtype=np.float32)
        scaled_vec = scaler.transform(raw_vec)[0]   # shape (n_features,)
        window.append(scaled_vec)

        # Only predict once we have a full window
        if len(window) < time_steps:
            log_rows.append({
                "step": step, "phase": phase,
                **{f: obs[f] for f in feat_cols},
                "risk_pct": None, "tier": "WARMING_UP", "warnings": "",
                "recommendation": "Collecting initial window …",
            })
            continue

        # Build sequence tensor
        seq = np.array(list(window), dtype=np.float32)     # (time_steps, n_features)
        seq = seq[np.newaxis, ...]                          # (1, time_steps, n_features)

        risk_prob = float(binary_model.predict(seq, verbose=0)[0][0])
        risk = score_risk(risk_prob, obs)

        # Print every N steps or when tier is RED
        should_print = verbose and (step % print_every == 0 or risk["tier"] == "RED")
        print_risk_report(step, phase, obs, risk, verbose=should_print)

        log_rows.append({
            "step": step,
            "phase": phase,
            **{f: round(obs[f], 3) for f in feat_cols},
            "risk_pct": risk["risk_pct"],
            "tier": risk["tier"],
            "warnings": " | ".join(risk["warnings"]),
            "recommendation": risk["recommendation"],
        })

    # Save CSV log
    df_log = pd.DataFrame(log_rows)
    df_log.to_csv(output_csv, index=False)
    print(f"\n{'═'*60}")
    print(f"  Simulation complete. Log saved → {output_csv}")

    # Summary stats
    valid = df_log[df_log["tier"].isin(["GREEN", "YELLOW", "RED"])]
    if len(valid):
        tc = valid["tier"].value_counts()
        print(f"\n  Risk tier breakdown:")
        for t in ["GREEN", "YELLOW", "RED"]:
            cnt = tc.get(t, 0)
            bar = "█" * int(cnt / len(valid) * 30)
            print(f"    {t:<8}: {cnt:>4} steps  {bar}")
    print("═" * 60)

    return df_log
