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
    get_plot_path,
    PLOTS_DIR,
)

os.makedirs(PLOTS_DIR, exist_ok=True)

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
    """Generates machine telemetry across 4 sequential degradation phases."""

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
        n_steps: int     = 200,
        phase_splits: tuple = PHASE_SPLITS,
        noise_std: float = NOISE_STD,
        random_seed: int = RANDOM_SEED,
    ):
        assert abs(sum(phase_splits) - 1.0) < 1e-6
        self.n_steps   = n_steps
        self.noise_std = noise_std
        np.random.seed(random_seed)

        cuts = np.cumsum([int(f * n_steps) for f in phase_splits])
        self.phase_ends = {
            "NORMAL":       cuts[0],
            "WARM-UP":      cuts[1],
            "STRESS":       cuts[2],
            "FAILURE-ZONE": n_steps,
        }
        self._state = {k: v for k, v in self.BASE.items()}
        self._step  = 0

    def _phase(self):
        s = self._step
        if s < self.phase_ends["NORMAL"]:       return "NORMAL"
        elif s < self.phase_ends["WARM-UP"]:    return "WARM-UP"
        elif s < self.phase_ends["STRESS"]:     return "STRESS"
        else:                                   return "FAILURE-ZONE"

    def next_step(self) -> dict:
        phase = self._phase()
        if phase == "WARM-UP":
            for k in self.DRIFT:
                self._state[k] += self.DRIFT[k] * 0.3
        elif phase == "STRESS":
            for k in self.DRIFT:
                self._state[k] += self.DRIFT[k]
        elif phase == "FAILURE-ZONE":
            for k in self.DRIFT:
                self._state[k] += self.DRIFT[k] * 2.5

        obs = {}
        for feat, val in self._state.items():
            if feat == "Type":
                obs[feat] = int(val)
            else:
                obs[feat] = max(0.0, val + np.random.normal(0, self.noise_std))

        self._step += 1
        obs["step"]  = self._step
        obs["phase"] = phase
        return obs

    def __iter__(self):
        for _ in range(self.n_steps):
            yield self.next_step()


# ── Risk scoring ──────────────────────────────────────────────────────────────

def score_risk(risk_prob: float, obs: dict) -> dict:
    """Compute tiered risk level + per-feature threshold warnings."""
    pct  = risk_prob * 100.0
    tier = "GREEN" if pct < 30 else "YELLOW" if pct < 70 else "RED"

    warnings = []
    for feat, threshold in FAULT_THRESHOLDS.items():
        val = obs.get(feat, 0)
        if feat == "Rotational speed [rpm]":
            if val < threshold:
                warnings.append(WARNING_MESSAGES[feat])
        else:
            if val > threshold:
                warnings.append(WARNING_MESSAGES[feat])

    return {
        "risk_pct":       round(pct, 2),
        "tier":           tier,
        "warnings":       warnings,
        "recommendation": RECOMMENDATIONS[tier],
    }


# ── Terminal reporter ─────────────────────────────────────────────────────────

def print_risk_report(step: int, phase: str, obs: dict, risk: dict, verbose: bool = True):
    if not verbose:
        return
    tier  = risk["tier"]
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
        print(f"  {color}[!] {w}{RESET}")
    print(f"  -> {risk['recommendation']}")


# ── Risk timeline plot ────────────────────────────────────────────────────────

def plot_risk_timeline(df_log: pd.DataFrame):
    """Save a color-coded risk % timeline as 09_Simulation_Risk_Timeline.png."""
    valid = df_log[df_log["tier"].isin(["GREEN", "YELLOW", "RED"])].copy()
    if valid.empty:
        return

    color_map = {"GREEN": "#2ecc71", "YELLOW": "#f39c12", "RED": "#e74c3c"}
    path = get_plot_path("sim_timeline")

    fig, ax = plt.subplots(figsize=(14, 4))
    for _, row in valid.iterrows():
        ax.bar(row["step"], row["risk_pct"],
               color=color_map.get(row["tier"], "#aaa"), width=1.0, linewidth=0)

    # Phase boundary markers
    phase_changes = valid[valid["phase"] != valid["phase"].shift()]
    for _, row in phase_changes.iterrows():
        ax.axvline(row["step"], color="#555", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.text(row["step"] + 0.5, 95, row["phase"], fontsize=7,
                color="#333", rotation=0, va="top")

    ax.axhline(30, color="#f39c12", linewidth=1, linestyle=":", alpha=0.7, label="Yellow threshold (30%)")
    ax.axhline(70, color="#e74c3c", linewidth=1, linestyle=":", alpha=0.7, label="Red threshold (70%)")
    ax.set_xlim(valid["step"].min(), valid["step"].max())
    ax.set_ylim(0, 105)
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Fault Risk (%)")
    ax.set_title("Robot Simulator — Fault Risk Timeline", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  [Saved] {path}")


# ── Main run function ─────────────────────────────────────────────────────────

def run_simulation(
    binary_model,
    scaler,
    time_steps: int  = 10,
    n_steps: int     = 200,
    output_csv: str  = "simulation_log.csv",
    verbose: bool    = True,
    print_every: int = 10,
):
    """Run the robot simulator and apply the trained binary LSTM at each step.

    Parameters
    ----------
    binary_model : trained Keras binary LSTM model
    scaler       : fitted StandardScaler from training
    time_steps   : sequence window length (must match training)
    n_steps      : total simulation steps
    output_csv   : path to save the step-by-step risk log CSV
    verbose      : whether to print live reports to terminal
    print_every  : print every N steps (reduces terminal noise)
    """
    print("\n" + "=" * 60)
    print("   ROBOT SIMULATOR — LIVE FAULT RISK SCORING")
    print("=" * 60)
    print(f"  Phases : NORMAL -> WARM-UP -> STRESS -> FAILURE-ZONE")
    print(f"  Steps  : {n_steps}")
    print(f"  Window : {time_steps} steps")
    print(f"  Log    : {output_csv}\n")

    sim      = RobotSimulator(n_steps=n_steps)
    feat_cols = [
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    ]
    window   = deque(maxlen=time_steps)
    log_rows = []

    for obs in sim:
        step  = obs["step"]
        phase = obs["phase"]

        raw_vec    = np.array([[obs[f] for f in feat_cols]], dtype=np.float32)
        scaled_vec = scaler.transform(raw_vec)[0]
        window.append(scaled_vec)

        if len(window) < time_steps:
            log_rows.append({
                "step": step, "phase": phase,
                **{f: obs[f] for f in feat_cols},
                "risk_pct": None, "tier": "WARMING_UP", "warnings": "",
                "recommendation": "Collecting initial window ...",
            })
            continue

        seq      = np.array(list(window), dtype=np.float32)[np.newaxis, ...]
        risk_prob = float(binary_model.predict(seq, verbose=0)[0][0])
        risk      = score_risk(risk_prob, obs)

        should_print = verbose and (step % print_every == 0 or risk["tier"] == "RED")
        print_risk_report(step, phase, obs, risk, verbose=should_print)

        log_rows.append({
            "step": step, "phase": phase,
            **{f: round(obs[f], 3) for f in feat_cols},
            "risk_pct":       risk["risk_pct"],
            "tier":           risk["tier"],
            "warnings":       " | ".join(risk["warnings"]),
            "recommendation": risk["recommendation"],
        })

    df_log = pd.DataFrame(log_rows)
    df_log.to_csv(output_csv, index=False)

    print(f"\n{'='*60}")
    print(f"  Simulation complete. Log saved -> {output_csv}")

    valid = df_log[df_log["tier"].isin(["GREEN", "YELLOW", "RED"])]
    if len(valid):
        tc = valid["tier"].value_counts()
        print(f"\n  Risk tier breakdown:")
        for t in ["GREEN", "YELLOW", "RED"]:
            cnt = tc.get(t, 0)
            bar = chr(9608) * int(cnt / len(valid) * 30)
            print(f"    {t:<8}: {cnt:>4} steps  {bar}")

    # Save timeline visualization
    plot_risk_timeline(df_log)
    print("=" * 60)

    return df_log
