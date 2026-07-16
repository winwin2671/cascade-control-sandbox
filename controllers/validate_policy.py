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
    python3 controllers/validate_policy.py --policy controllers/sac_threetank.zip
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
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--control-dt", type=float, default=0.5)
    args = ap.parse_args()

    from stable_baselines3 import SAC, PPO  # noqa: E402
    from aio_bridge_env import CascadeBridgeEnv  # noqa: E402

    cls = SAC if args.algo == "sac" else PPO
    model = cls.load(args.policy)
    LOG.info("loaded policy: %s (%s)", args.policy, args.algo)

    # B2 fix: detect action mode from the metadata sidecar
    import json
    meta_path = args.policy.replace(".zip", ".json")
    if Path(meta_path).exists():
        meta = json.load(open(meta_path))
        if meta.get("action_mode") == "setpoint":
            LOG.warning("Policy trained in setpoint mode — use `./run_mode.sh rl` instead "
                        "(run_rl.py handles setpoint → *_sp). This validator assumes actuator mode.")

    plant = ThreeTankModel()
    scorer = KPIScorer(plant)
    h_sp_dict, t_sp = plant.default_setpoints()       # {0:0.45,1:0.30,2:0.40}, [45,45,45]
    h_sp_ctrl = [h_sp_dict[i] for i in plant.controlled_levels()]  # [0.45, 0.40]
    h_sp_all = [h_sp_dict.get(i, 0.0) for i in range(plant.n)]     # [0.45, 0.30, 0.40]
    t_cold, t_amb = plant.t_supply, plant.t_ambient

    env = CascadeBridgeEnv(backend="ia2", control_dt=args.control_dt, mode="mpc")

    obs, _ = env.reset()
    scorer.reset()
    rewards = []
    for k in range(args.steps):
        full_obs = ia2_to_aiogym_obs(obs, h_sp_ctrl, t_sp, t_cold, t_amb)
        action, _ = model.predict(full_obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float32).flatten(), 0.0, 1.0)
        obs, reward, _, _, info = env.step(action)
        rewards.append(reward)

        levels = [float(obs[0]), float(obs[2]), float(obs[4])]
        temps = [float(obs[1]), float(obs[3]), float(obs[5])]
        act_dict = {"pumps": list(action[:2]), "valves": [], "heaters": list(action[2:])}
        heat_w = plant.heater_power(act_dict)
        ideal_w = plant.ideal_power(levels, temps, t_sp,
                                    {"t_cold": t_cold, "t_amb": t_amb}, act_dict)
        scorer.step_penalty(levels, temps, h_sp_all, t_sp, heat_w, ideal_w, False,
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
