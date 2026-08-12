"""Validation gate — validate a trained RL policy on the IA2 track (sim-to-real).

Loads a trained SB3 policy (.zip), runs it through the IA2-in-the-loop track
(CascadeBridgeEnv backend=ia2 — real 50 ms scan + iomap + the L5 safety shield),
computes the same KPI the benchmark uses (via KPIScorer), and reports the sim-to-
real gap.

Handles BOTH action modes:
  --action-mode actuator (default): policy outputs the 5 MVs in contract order
      [v_12, v_23, e_101, v_33, vfd] -> env.step writes them directly (PLC mode=mpc).
  --action-mode setpoint: policy outputs supervisory setpoints (numpy-track setpoint
      policy) -> writes the PID *_sp vars (PLC mode=pid), PID tracks. Requires a
      PLC backend (ia2/edge).

The obs the policy expects depends on the training track; we reconstruct it from
the 14-dim bridge obs + config setpoints (see _build_obs).

Requires the IA2 chain up: mock_cabinet.py + ia2-server + cs project open + cs run.

Usage:
    python3 controllers/validate_policy.py --policy controllers/policies/sac_threetank.zip --backend ia2
    python3 controllers/validate_policy.py --policy ... --action-mode setpoint --backend ia2

Note on edge backend: each step needs an SSH round-trip proxied through the dev
server (~6 handshakes per 0.5 s step) — raise --control-dt for edge deployments.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from controllers.aiogym_register import register_threetank  # noqa: E402
register_threetank()
from controllers.threetank_model import ThreeTankModel  # noqa: E402
from aiogym.scoring import KPIScorer  # noqa: E402

LOG = logging.getLogger("validate")


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("pymodbus").setLevel(logging.WARNING)

    ap = argparse.ArgumentParser(description="Validation gate — RL policy on IA2 (sim-to-real).")
    ap.add_argument("--policy", required=True, help="path to the SB3 .zip policy")
    ap.add_argument("--algo", default="sac", choices=["sac", "ppo"])
    ap.add_argument("--backend", default="ia2",
                    help="Communication backend: auto | ia2 | modbus | edge:<name> (default: ia2)")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--control-dt", type=float, default=0.5)
    ap.add_argument("--action-mode", default=None, choices=["actuator", "setpoint"],
                    help="Override the policy action mode (otherwise read from the .json "
                         "sidecar written by train_sb3.py). Required when no sidecar is present.")
    args = ap.parse_args()

    from stable_baselines3 import SAC, PPO  # noqa: E402
    from aio_bridge_env import CascadeBridgeEnv  # noqa: E402

    cls = SAC if args.algo == "sac" else PPO
    model = cls.load(args.policy)
    LOG.info("loaded policy: %s (%s)", args.policy, args.algo)

    # Resolve action mode: --action-mode flag > metadata sidecar > hard error.
    meta_path = args.policy.replace(".zip", ".json")
    if args.action_mode is not None:
        action_mode = args.action_mode
    elif Path(meta_path).exists():
        action_mode = json.load(open(meta_path)).get("action_mode")
        if action_mode not in ("actuator", "setpoint"):
            LOG.error("Metadata sidecar %s has action_mode=%r; must be 'actuator' or 'setpoint'. "
                      "Re-train or pass --action-mode.", meta_path, action_mode)
            sys.exit(1)
    else:
        LOG.error("Cannot determine action mode: no sidecar at %s and no --action-mode given.",
                  meta_path)
        sys.exit(1)
    LOG.info("action_mode: %s", action_mode)

    plant = ThreeTankModel()
    scorer = KPIScorer(plant)
    h_sp_dict, t_sp = plant.default_setpoints()
    h_sp_all = [h_sp_dict.get(i, 0.0) for i in range(plant.n)]
    t_cold, t_amb = plant.t_supply, plant.t_ambient

    plc_mode = "pid" if action_mode == "setpoint" else "mpc"
    env = CascadeBridgeEnv(backend=args.backend, control_dt=args.control_dt, mode=plc_mode)
    b = env.backend
    # setpoint mode writes PLC-internal *_sp vars -> needs a PLC backend (ia2/edge).
    if action_mode == "setpoint" and not b.writes_via_plc:
        LOG.error("setpoint-mode policy requires a PLC backend (ia2/edge); the %s backend has "
                  "no *_sp registers. Use --backend ia2, or validate an actuator-mode policy.",
                  args.backend)
        sys.exit(1)

    obs, _ = env.reset()
    expected_shape = model.observation_space.shape
    n_raw = env.observation_space.shape[0]                       # 14 sensors
    am = env._act_max                                            # per-actuator eng max (->fraction)
    # context for the EnrichedObs (modbus-track) policy: 3 level sp + 3 temp sp + t_cold + t_amb
    ctx_enriched = [h_sp_dict[i] for i in range(plant.n)] + list(t_sp) + [t_cold, t_amb]
    # numpy-track setpoint/actuator obs reconstruction: [levels(3), temps(3), t_sp(3), h_sp(3), t_cold, t_amb]
    def numpy_obs():
        levels = [float(obs[0]), float(obs[2]), float(obs[4])]
        temps = [float(obs[1]), float(obs[3]), float(obs[5])]
        return np.array(levels + temps + list(t_sp) + h_sp_all + [t_cold, t_amb], dtype=np.float32)

    scorer.reset()
    rewards = []
    for k in range(args.steps):
        # build the obs the policy expects
        if expected_shape == (n_raw + len(ctx_enriched),):      # EnrichedObs (train_rl modbus)
            full_obs = np.concatenate([obs, np.array(ctx_enriched, dtype=np.float32)])
        elif expected_shape == (n_raw,):                        # raw 14-dim bridge obs
            full_obs = np.asarray(obs, dtype=np.float32)
        else:                                                   # numpy-track obs (14-dim reconstruction)
            full_obs = numpy_obs()
        action, _ = model.predict(full_obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float32).flatten(), 0.0, 1.0)

        if action_mode == "setpoint":
            # numpy setpoint policy: action = [t_sp0,t_sp1,t_sp2, h_sp0,h_sp1,h_sp2] (SUPERVISORY).
            # Write the PID-controlled sp vars: 3 level_sp + tank1_temp_sp (only T1 has a temp PID).
            for i, name in enumerate(("tank1_level_sp", "tank2_level_sp", "tank3_level_sp")):
                b.write_register(name, 0.15 + action[3 + i] * 0.30)      # 0.15..0.45 m
            b.write_register("tank1_temp_sp", 20.0 + action[0] * 60.0)   # 20..80 degC
            time.sleep(args.control_dt)
            raw_vars = b.read_raw()
            obs = env._decode_obs(raw_vars)
            sidx = {n: i for i, n in enumerate(env.sensor_names)}
            levels_d = {n: float(obs[sidx[n]]) for n in env.setpoints}
            temps_d = {n: float(obs[sidx[n]]) for n in env.temp_setpoints}
            track_l = sum((levels_d[n] - env.setpoints[n]) ** 2 for n in env.setpoints)
            track_t = sum((temps_d[n] - env.temp_setpoints[n]) ** 2 for n in env.temp_setpoints)
            w = env.reward_weights
            reward = float(-(w["level"] * track_l + w["temp"] * track_t))
            info = {"levels_m": levels_d, "temps_c": temps_d, "raw": raw_vars}
        else:
            obs, reward, _, _, info = env.step(action)          # contract-order 5-MV; env writes directly
        rewards.append(reward)

        levels = [float(obs[0]), float(obs[2]), float(obs[4])]
        temps = [float(obs[1]), float(obs[3]), float(obs[5])]
        # KPI energy term: applied (post-L5) actuator values -> model fractions.
        raw = info["raw"]
        act_dict = {
            "pumps":   [raw.get("vfd_cmd", 0.0) / am["vfd_cmd"]],
            "valves":  [raw.get("v_12_cmd", 0.0) / am["v_12_cmd"],
                        raw.get("v_23_cmd", 0.0) / am["v_23_cmd"],
                        raw.get("v_33_cmd", 0.0) / am["v_33_cmd"]],
            "heaters": [raw.get("e_101_cmd", 0.0) / am["e_101_cmd"]],
        }
        heat_w = plant.heater_power(act_dict)
        ideal_w = plant.ideal_power(levels, temps, t_sp,
                                    {"t_cold": t_cold, "t_amb": t_amb}, act_dict)
        from controllers.rollout_report import detect_interlock
        interlock = detect_interlock(info["raw"])
        scorer.step_penalty(levels, temps, h_sp_all, t_sp, heat_w, ideal_w, interlock,
                            args.control_dt)
        if k % 10 == 0 or k == args.steps - 1:
            lv, tp = info["levels_m"], info.get("temps_c", {})
            LOG.info("step %2d  levels(m)=%.3f/%.3f/%.3f  temps(C)=%.1f/%.1f/%.1f  r=%.3f",
                     k, lv.get("tank1_level", float("nan")), lv.get("tank2_level", float("nan")),
                     lv.get("tank3_level", float("nan")), tp.get("tank1_temp", float("nan")),
                     tp.get("tank2_temp", float("nan")), tp.get("tank3_temp", float("nan")), reward)
    env.close()

    kpi = scorer.report()
    LOG.info("=== Validation gate (IA2 track, %d steps, action_mode=%s) ===", args.steps, action_mode)
    LOG.info("KPI score: %.2f", kpi["score"])
    LOG.info("  temp_err=%.2f  level_err_cm=%.2f  excess_kwh=%.3f  interlock=%.2f",
             kpi["avg_temp_err"], kpi["avg_level_err_cm"], kpi["excess_kwh"], kpi["interlock_frac"])
    LOG.info("  mean reward: %.4f", float(np.mean(rewards)))
    LOG.info("Compare to the numpy-env benchmark (controllers/benchmark.py) for the sim-to-real gap.")


if __name__ == "__main__":
    main()
