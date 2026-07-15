"""Shared rollout report — KPI table + CSV + PNG plot for any controller run.

Each controller (run_mpc.py, run_rl.py, etc.) collects step-by-step data during
its rollout, then calls report(steps_data, tag) at the end. This utility:
  1. Computes the KPI via AIO-Gym's KPIScorer (score, temp_err, level_err, etc.)
  2. Prints an AIO-Gym-style KPI table to the terminal
  3. Saves a CSV (step, levels, temps, actions, reward)
  4. Saves a 3-panel PNG plot (levels, temps, reward with setpoint lines)

Usage:
    from controllers.rollout_report import report
    steps_data = []   # collect during the rollout
    report(steps_data, tag="mpc")
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
AIO_GYM = str(Path.home() / "projects" / "AIO-Gym")
if AIO_GYM not in sys.path:
    sys.path.insert(0, AIO_GYM)


def report(steps_data: list[dict], tag: str = "rollout",
           out_dir: Path | str | None = None,
           control_dt: float = 0.5) -> dict:
    """Print KPI table + save CSV + save PNG.

    Each entry in steps_data must have:
        step (int), levels (list[3] m), temps (list[3] degC),
        action (list[5] 0-1), reward (float)
    Returns the scorer.report() dict.
    """
    out_dir = Path(out_dir or ROOT / "controllers" / "runs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- KPI via AIO-Gym's scorer ---
    from controllers.aiogym_register import register_threetank
    register_threetank()
    from controllers.threetank_model import ThreeTankModel
    from aiogym.scoring import KPIScorer

    model = ThreeTankModel()
    cfg = json.load(open(ROOT / "ia2_config.json"))
    hsp = cfg["control"]["setpoints_m"]
    h_sp = [hsp["tank1_level"], hsp["tank2_level"], hsp["tank3_level"]]
    t_sp = list(cfg["control"]["setpoints_c"].values())
    t_cold = float(cfg["process"]["t_supply_c"])
    t_amb = float(cfg["process"]["t_ambient_c"])

    scorer = KPIScorer(model)
    scorer.reset()

    for sd in steps_data:
        levels = sd["levels"]
        temps = sd["temps"]
        act = {"pumps": list(sd["action"][:2]), "valves": [],
               "heaters": list(sd["action"][2:])}
        env_dict = {"t_cold": t_cold, "t_amb": t_amb, "extra_outflow": 0.0}
        heat_w = model.heater_power(act)
        ideal_w = model.ideal_power(levels, temps, t_sp, env_dict, act)
        scorer.step_penalty(levels, temps, h_sp, t_sp,
                             heat_w, ideal_w, False, control_dt)

    rep = scorer.report()
    mean_reward = float(np.mean([sd["reward"] for sd in steps_data]))

    # --- print KPI table ---
    print(f"\n=== KPI Report ({len(steps_data)} steps, {tag}) ===")
    print(f"  score:       {rep['score']:6.1f}   (out of 100)")
    print(f"  temp_err:    {rep['avg_temp_err']:6.1f} °C (avg)")
    print(f"  level_err:   {rep['avg_level_err_cm']:6.1f} cm (avg)")
    print(f"  excess_kwh:  {rep['excess_kwh']:6.3f}")
    print(f"  interlock:   {rep['interlock_frac'] * 100:5.1f}%")
    print(f"  mean reward: {mean_reward:8.4f}")

    # --- save CSV ---
    csv_path = out_dir / f"{tag}_rollout.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "h1", "h2", "h3", "T1", "T2", "T3",
                     "act1", "act2", "act3", "act4", "act5", "reward"])
        for sd in steps_data:
            w.writerow([sd["step"], *sd["levels"], *sd["temps"],
                        *sd["action"], sd["reward"]])
    print(f"\n  saved: {csv_path}")

    # --- save PNG ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [sd["step"] for sd in steps_data]
        H = np.array([[sd["levels"][j] for sd in steps_data] for j in range(3)]).T
        T_arr = np.array([[sd["temps"][j] for sd in steps_data] for j in range(3)]).T
        R = [sd["reward"] for sd in steps_data]
        colors = ["#2196F3", "#4CAF50", "#FF9800"]

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        # levels + setpoints
        for j in range(3):
            axes[0].plot(steps, H[:, j], color=colors[j], label=f"h{j + 1}")
            if j < len(h_sp):
                axes[0].axhline(h_sp[j], color=colors[j], ls="--", alpha=0.4)
        axes[0].set_ylabel("Level (m)")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].set_title(f"{tag} — KPI {rep['score']:.1f}", fontsize=12)
        # temps + setpoint
        for j in range(3):
            axes[1].plot(steps, T_arr[:, j], color=colors[j], label=f"T{j + 1}")
        axes[1].axhline(t_sp[0], color="gray", ls="--", alpha=0.4, label="SP")
        axes[1].set_ylabel("Temp (°C)")
        axes[1].legend(loc="upper right", fontsize=8)
        # reward
        axes[2].plot(steps, R, "#E91E63", label="reward")
        axes[2].axhline(0, color="gray", ls=":", alpha=0.3)
        axes[2].set_ylabel("Reward")
        axes[2].set_xlabel("Step")
        axes[2].legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        png_path = out_dir / f"{tag}_rollout.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  saved: {png_path}")
    except ImportError:
        print("  (matplotlib not installed — skipping PNG)")

    return rep
