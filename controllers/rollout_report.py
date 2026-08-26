"""Shared rollout report — KPI table + CSV + PNG plot for any controller run.

Each controller (run_mpc.py, run_rl.py, etc.) collects step-by-step data during
its rollout, then calls report(steps_data, tag) at the end. This utility:
  1. Computes the KPI via AIO-Gym's KPIScorer (score, temp_err, level_err, etc.)
  2. Prints an AIO-Gym-style KPI table to the terminal
  3. Saves a CSV (step, levels, temps, actions [+ reward for RL runs])
  4. Saves a PNG plot (levels, temps [+ reward panel for RL runs])

The per-step "reward" key is OPTIONAL and RL-only — PID/MPC/NMPC/Manual don't
optimize a reward, so their runs omit it and the report drops the mean-reward
line, the CSV column, and the reward panel accordingly.

Usage:
    from controllers.rollout_report import report, detect_interlock
    steps_data = []   # collect during the rollout
    report(steps_data, tag="mpc")
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def detect_interlock(raw_snapshot: dict) -> bool:
    """C4 fix: detect if the L5 software shield intervened this step.

    Compares the pre-shield request vars (*_req) to the post-shield mapped
    outputs. If a request is nonzero but the mapped output is 0, the shield
    forced a cutoff. Works for MPC/RL-actuator modes (where *_req is populated).
    For PID/Manual/setpoint modes, *_req is 0 (the PID output doesn't go through
    the req path), so this returns False — shield interventions in those modes
    are not detected (the PID's intermediate output isn't in the snapshot).
    """
    for req_name, mapped_name in [
        ("vfd_cmd_req", "vfd_cmd"), ("v_12_cmd_req", "v_12_cmd"),
        ("v_23_cmd_req", "v_23_cmd"), ("v_33_cmd_req", "v_33_cmd"),
        ("e_101_cmd_req", "e_101_cmd"),
    ]:
        req = raw_snapshot.get(req_name, 0)
        mapped = raw_snapshot.get(mapped_name, 0)
        if req > 0 and mapped == 0:
            return True
    return False


def report(steps_data: list[dict], tag: str = "rollout",
           out_dir: Path | str | None = None,
           control_dt: float = 0.5) -> dict:
    """Print KPI table + save CSV + save PNG.

    Each entry in steps_data must have:
        step (int), levels (list[3] m), temps (list[3] degC),
        action (list[5] 0-1)
    Optional per-step keys:
        applied_duty (list[5], 0-1) — post-L5-shield actuator/heater duty used for
            the energy KPI when `action` is NOT the applied duty (e.g. setpoint-
            mode policies, where `action` holds normalized setpoints, not duties).
            If absent, `action` is used (correct for actuator/mpc/nmpc/manual
            callers, which already log the true duty as `action`).
        interlock (bool)
    Returns the scorer.report() dict.
    """
    out_dir = Path(out_dir or ROOT / "controllers" / "runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    # --disturbance runs (run_mode.sh exports DISTURBANCE=true when the fault
    # sidecar is active) suffix the tag, so fault-injection artifacts land in
    # <tag>_dist_rollout.csv/png instead of overwriting the baseline rollout.
    if os.environ.get("DISTURBANCE", "").lower() == "true":
        tag = f"{tag}_dist"

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
        # Energy KPI uses the *applied* duty (post-L5-shield) when the caller
        # supplies it; otherwise fall back to the logged action. Setpoint-mode
        # policies must supply applied_duty — their `action` holds normalized
        # setpoints, so feeding it to heater_power() as duty produces a meaningless
        # excess_kwh (same defect as validate_policy.py:147-165, fixed by reading
        # back the actuator*/heater* registers).
        # duty = applied (post-L5) actuator FRACTIONS in contract order
        # [v_12, v_23, e_101, v_33, vfd]; map to the model's pumps/valves/heaters.
        duty = sd.get("applied_duty", sd["action"])
        dmap = dict(zip(["v_12_cmd", "v_23_cmd", "e_101_cmd", "v_33_cmd", "vfd_cmd"], duty))
        act = {"pumps": [dmap["vfd_cmd"]],
               "valves": [dmap["v_12_cmd"], dmap["v_23_cmd"], dmap["v_33_cmd"]],
               "heaters": [dmap["e_101_cmd"]]}
        env_dict = {"t_cold": t_cold, "t_amb": t_amb}
        heat_w = model.heater_power(act)
        ideal_w = model.ideal_power(levels, temps, t_sp, env_dict, act)
        scorer.step_penalty(levels, temps, h_sp, t_sp,
                             heat_w, ideal_w, sd.get("interlock", False), control_dt)

    rep = scorer.report()
    # RL-only: the per-step reward key is absent for PID/MPC/NMPC/Manual runs.
    has_reward = bool(steps_data) and all("reward" in sd for sd in steps_data)
    mean_reward = float(np.mean([sd["reward"] for sd in steps_data])) if has_reward else None

    # --- print KPI table ---
    print(f"\n=== KPI Report ({len(steps_data)} steps, {tag}) ===")
    print(f"  score:       {rep['score']:6.1f}   (out of 100)")
    print(f"  temp_err:    {rep['avg_temp_err']:6.1f} °C (avg)")
    print(f"  level_err:   {rep['avg_level_err_cm']:6.1f} cm (avg)")
    print(f"  excess_kwh:  {rep['excess_kwh']:6.3f}")
    print(f"  interlock:   {rep['interlock_frac'] * 100:5.1f}%")
    if has_reward:
        print(f"  mean reward: {mean_reward:8.4f}")

    # --- save CSV ---
    csv_path = out_dir / f"{tag}_rollout.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["step", "h1", "h2", "h3", "T1", "T2", "T3",
                  "act1", "act2", "act3", "act4", "act5"]
        if has_reward:
            header.append("reward")
        w.writerow(header)
        for sd in steps_data:
            row = [sd["step"], *sd["levels"], *sd["temps"], *sd["action"]]
            if has_reward:
                row.append(sd["reward"])
            w.writerow(row)
    print(f"\n  saved: {csv_path}")

    # --- save PNG ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [sd["step"] for sd in steps_data]
        H = np.array([[sd["levels"][j] for sd in steps_data] for j in range(3)]).T
        T_arr = np.array([[sd["temps"][j] for sd in steps_data] for j in range(3)]).T
        colors = ["#2196F3", "#4CAF50", "#FF9800"]

        fig, axes = plt.subplots(3 if has_reward else 2, 1,
                                 figsize=(10, 8 if has_reward else 6), sharex=True)
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
        if has_reward:
            # reward (RL runs only)
            R = [sd["reward"] for sd in steps_data]
            axes[2].plot(steps, R, "#E91E63", label="reward")
            axes[2].axhline(0, color="gray", ls=":", alpha=0.3)
            axes[2].set_ylabel("Reward")
            axes[2].set_xlabel("Step")
            axes[2].legend(loc="upper right", fontsize=8)
        else:
            axes[-1].set_xlabel("Step")

        plt.tight_layout()
        png_path = out_dir / f"{tag}_rollout.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  saved: {png_path}")
    except ImportError:
        print("  (matplotlib not installed — skipping PNG)")

    return rep
