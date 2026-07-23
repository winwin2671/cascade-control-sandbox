"""Validation gate — validate a trained RL policy on the IA2 track (sim-to-real).

Loads a trained SB3 policy (.zip), runs it through the IA2-in-the-loop track
(CascadeBridgeEnv backend=ia2 — real 50 ms scan + iomap + the L5 safety shield),
computes the same KPI the benchmark uses (via KPIScorer), and reports the sim-to-
real gap.

The policy was trained on AIOGymNativeEnv's 13-dim obs
[levels(3), temps(3), t_sp(3), h_sp_ctrl(2), t_cold, t_amb]; the IA2 env gives a
6-dim obs [h1,T1,h2,T2,h3,T3] — we reconstruct the 13-dim obs using the config
setpoints. The policy outputs a 5-dim action [p1,p2,h1,h2,h3] → written to
actuator*_req (mode=mpc) → through the L5 shield → to the cabinet.

Requires the IA2 chain up: mock_cabinet.py + ia2-server + cs project open + cs run.

Usage:
    python3 controllers/validate_policy.py --policy controllers/policies/sac_threetank.zip --backend ia2

Note on edge backend: If using `--backend edge:<name>`, be aware that each
step requires an SSH round-trip proxied through the dev server (~6 handshakes
per 0.5 s step). For edge deployments, increase `--control-dt` to accommodate
the network latency.
"""
from __future__ import annotations

import argparse
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


def ia2_to_aiogym_obs(ia2_obs, h_sp_ctrl, t_sp, t_cold, t_amb):
    """Reconstruct the 13-dim AIOGymNativeEnv obs from the 6-dim IA2 obs.

    obs = [levels(3), temps(3), t_sp(3), h_sp_ctrl(2), t_cold, t_amb].
    h_sp_ctrl = the controlled-level setpoints (indices [0,2] for our plant).
    """
    levels = [float(ia2_obs[0]), float(ia2_obs[2]), float(ia2_obs[4])]
    temps = [float(ia2_obs[1]), float(ia2_obs[3]), float(ia2_obs[5])]
    return np.array(levels + temps + list(t_sp) + list(h_sp_ctrl) + [t_cold, t_amb],
                    dtype=np.float32)


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
                    help="Override the policy action mode (otherwise read from the "
                         ".json sidecar written by train_sb3.py). Required when no "
                         "sidecar is present.")
    args = ap.parse_args()

    from stable_baselines3 import SAC, PPO  # noqa: E402
    from aio_bridge_env import CascadeBridgeEnv  # noqa: E402

    cls = SAC if args.algo == "sac" else PPO
    model = cls.load(args.policy)
    LOG.info("loaded policy: %s (%s)", args.policy, args.algo)

    # B2/B2c fix: dispatch on action mode. Priority: --action-mode flag >
    # metadata sidecar > hard error. A missing sidecar is an ERROR (B2c), not a
    # silent actuator assumption — a setpoint policy whose sidecar was lost would
    # otherwise be scored as actuator garbage with only a log line (the exact
    # original failure). This guard runs before env construction, so the error is
    # reachable without the IA2 chain up.
    import json
    meta_path = args.policy.replace(".zip", ".json")
    if args.action_mode is not None:
        action_mode = args.action_mode
        if Path(meta_path).exists():
            sidecar_mode = json.load(open(meta_path)).get("action_mode")
            if sidecar_mode and sidecar_mode != action_mode:
                LOG.warning("sidecar %s says action_mode=%s but --action-mode=%s "
                            "given; using the CLI value.", meta_path, sidecar_mode, action_mode)
    elif Path(meta_path).exists():
        action_mode = json.load(open(meta_path)).get("action_mode")
        if action_mode is None:
            LOG.error("Metadata sidecar %s has no action_mode key. Pass "
                      "--action-mode {actuator,setpoint} explicitly.", meta_path)
            sys.exit(1)
    else:
        LOG.error("Cannot determine action mode: no metadata sidecar at %s and no "
                  "--action-mode given. Re-train with train_sb3.py (it writes the "
                  ".json sidecar) or pass --action-mode {actuator,setpoint}.", meta_path)
        sys.exit(1)
    LOG.info("action_mode: %s", action_mode)

    plant = ThreeTankModel()
    scorer = KPIScorer(plant)
    h_sp_dict, t_sp = plant.default_setpoints()
    h_sp_ctrl = [h_sp_dict[i] for i in plant.controlled_levels()]
    h_sp_all = [h_sp_dict.get(i, 0.0) for i in range(plant.n)]
    t_cold, t_amb = plant.t_supply, plant.t_ambient

    # Setpoint mode → PID mode (PLC tracks setpoints); actuator mode → MPC mode (direct)
    plc_mode = "pid" if action_mode == "setpoint" else "mpc"
    env = CascadeBridgeEnv(backend=args.backend, control_dt=args.control_dt, mode=plc_mode)
    b = env.backend

    # Setpoint mode writes PLC variables (*_sp) that only exist on the IA2/edge
    # backends; a direct-Modbus backend has no such registers and write_register
    # would raise a cryptic ValueError mid-rollout. Reject up front (same class
    # of guard as manual_gui.py). Checking writes_via_plc also covers --backend
    # auto once it has resolved to a concrete backend.
    if action_mode == "setpoint" and not b.writes_via_plc:
        LOG.error("setpoint-mode policy requires a PLC backend (ia2/edge); the %s "
                  "backend has no *_sp registers. Use --backend ia2, or validate "
                  "an actuator-mode policy on modbus.", args.backend)
        sys.exit(1)

    obs, _ = env.reset()
    scorer.reset()
    rewards = []
    for k in range(args.steps):
        full_obs = ia2_to_aiogym_obs(obs, h_sp_ctrl, t_sp, t_cold, t_amb)
        action, _ = model.predict(full_obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float32).flatten(), 0.0, 1.0)

        if action_mode == "setpoint":
            # Map [t_sp0,t_sp1,t_sp2,h_sp0,h_sp2] to *_sp vars (same as run_rl.py)
            t0 = 20.0 + action[0] * 60.0; t1 = 20.0 + action[1] * 60.0; t2 = 20.0 + action[2] * 60.0
            h0 = 0.15 + action[3] * 0.40; h2 = 0.15 + action[4] * 0.40
            b.write_register("tank1_temp_sp", int(round(t0 / 0.01)))
            b.write_register("tank2_temp_sp", int(round(t1 / 0.01)))
            b.write_register("tank3_temp_sp", int(round(t2 / 0.01)))
            b.write_register("tank1_level_sp", int(round(h0 / 0.0001)))
            b.write_register("tank3_level_sp", int(round(h2 / 0.0001)))
            import time as _time
            _time.sleep(args.control_dt)
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
            obs, reward, _, _, info = env.step(action)
        rewards.append(reward)

        levels = [float(obs[0]), float(obs[2]), float(obs[4])]
        temps = [float(obs[1]), float(obs[3]), float(obs[5])]
        # Energy KPI must use the *applied* actuator/heater duty (post-L5-shield),
        # NOT the raw policy action: in setpoint mode `action` is [t_sp, h_sp]
        # setpoints, so feeding it to heater_power() as duty produces a meaningless
        # excess_kwh. The applied duty lives in the actuator*/heater* registers
        # (raw 0..10000 = 0..1). Reading them back also makes the energy accounting
        # shield-aware in actuator mode (a clamped request costs no power).
        raw = info["raw"]
        act_dict = {
            "pumps": [raw.get("actuator1", 0) * 1e-4, raw.get("actuator2", 0) * 1e-4],
            "valves": [],
            "heaters": [raw.get(n, 0) * 1e-4 for n in ("heater1", "heater2", "heater3")],
        }
        heat_w = plant.heater_power(act_dict)
        ideal_w = plant.ideal_power(levels, temps, t_sp,
                                    {"t_cold": t_cold, "t_amb": t_amb}, act_dict)
        from controllers.rollout_report import detect_interlock
        interlock = detect_interlock(info["raw"])  # R1 fix: use stashed read
        scorer.step_penalty(levels, temps, h_sp_all, t_sp, heat_w, ideal_w, interlock,
                            args.control_dt)
        if k % 10 == 0 or k == args.steps - 1:
            lv, tp = info["levels_m"], info.get("temps_c", {})
            LOG.info("step %2d  levels(m)=%.3f/%.3f/%.3f  temps(C)=%.1f/%.1f/%.1f  r=%.3f",
                     k, lv.get("tank1_level", float("nan")),
                     lv.get("tank2_level", float("nan")),
                     lv.get("tank3_level", float("nan")),
                     tp.get("tank1_temp", float("nan")),
                     tp.get("tank2_temp", float("nan")),
                     tp.get("tank3_temp", float("nan")), reward)
    env.close()

    kpi = scorer.report()
    LOG.info("=== Validation gate (IA2 track, %d steps) ===", args.steps)
    LOG.info("KPI score: %.2f", kpi["score"])
    LOG.info("  temp_err=%.2f  level_err_cm=%.2f  excess_kwh=%.3f  interlock=%.2f",
             kpi["avg_temp_err"], kpi["avg_level_err_cm"], kpi["excess_kwh"],
             kpi["interlock_frac"])
    LOG.info("  mean reward: %.4f", float(np.mean(rewards)))
    LOG.info("Compare to the numpy-env benchmark (controllers/benchmark.py) for the sim-to-real gap.")


if __name__ == "__main__":
    main()