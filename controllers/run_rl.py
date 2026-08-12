"""RL supervisor — drives the live plant with a trained SAC/PPO policy.

Like run_mpc.py / run_nmpc.py but the controller is a trained RL policy. Loads the
.zip, runs it through the IA2 track or directly via Modbus, reports
levels/temps/reward per step.

Handles BOTH action modes:
  --action-mode actuator (default): policy outputs the 5 MVs in contract order
      [v_12, v_23, e_101, v_33, vfd] -> env.step writes them directly (PLC mode=rl).
  --action-mode setpoint: policy outputs supervisory setpoints (numpy-track setpoint
      policy) -> writes the PID *_sp vars (PLC mode=pid), PID tracks. Requires a
      PLC backend (ia2/edge).

The obs the policy expects depends on the training track (EnrichedObs 22-dim,
raw 14-dim, or numpy 14-dim reconstruction).

Usage:
    python3 controllers/run_rl.py --policy controllers/policies/sac_threetank.zip --backend modbus
    python3 controllers/run_rl.py --backend ia2 --action-mode setpoint

Requires either:
    1) IA2 chain up: mock_cabinet.py + ia2-server + `cs project open` + `cs run`
    2) Modbus track: mock_cabinet.py (direct control, no PLC logic)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aio_bridge_env import CascadeBridgeEnv  # noqa: E402
from controllers.rollout_report import report, detect_interlock  # noqa: E402

LOG = logging.getLogger("rl_supervisor")


def main():
    ap = argparse.ArgumentParser(description="RL supervisor — trained policy on the IA2 or Modbus track.")
    ap.add_argument("--policy", default=str(ROOT / "controllers" / "policies" / "sac_threetank.zip"))
    ap.add_argument("--backend", default="ia2",
                    help="Communication backend: auto | ia2 | modbus | edge:<name> (default: ia2)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--control-dt", type=float, default=0.5)
    ap.add_argument("--action-mode", default=None, choices=["actuator", "setpoint"],
                    help="Override the policy action mode (otherwise read from the .json sidecar).")
    args = ap.parse_args()

    # Resolve action mode: --action-mode flag > metadata sidecar > default actuator.
    meta_path = args.policy.replace(".zip", ".json")
    if args.action_mode is not None:
        action_mode = args.action_mode
    elif Path(meta_path).exists():
        action_mode = json.load(open(meta_path)).get("action_mode", "actuator")
        if action_mode not in ("actuator", "setpoint"):
            action_mode = "actuator"
    else:
        action_mode = "actuator"

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("pymodbus").setLevel(logging.WARNING)

    from stable_baselines3 import SAC, PPO
    try:
        model = SAC.load(args.policy)
    except Exception:
        model = PPO.load(args.policy)
    LOG.info("loaded policy: %s  action_mode: %s  backend: %s",
             args.policy, action_mode, args.backend)

    cfg = json.load(open(ROOT / "ia2_config.json"))
    hsp = cfg["control"]["setpoints_m"]
    tsp = list(cfg["control"]["setpoints_c"].values())
    t_cold = float(cfg["process"]["t_supply_c"])
    t_amb = float(cfg["process"]["t_ambient_c"])

    plc_mode = "pid" if action_mode == "setpoint" else "rl"
    env = CascadeBridgeEnv(backend=args.backend, control_dt=args.control_dt, mode=plc_mode)
    b = env.backend
    if action_mode == "setpoint" and not b.writes_via_plc:
        LOG.error("setpoint-mode policy requires a PLC backend (ia2/edge); the %s backend has "
                  "no *_sp registers. Use --backend ia2, or an actuator-mode policy on modbus.",
                  args.backend)
        sys.exit(1)

    obs, _ = env.reset()
    LOG.info("RL supervisor start — action_mode=%s", action_mode)
    rewards, steps_data = [], []
    expected_shape = model.observation_space.shape
    n_raw = env.observation_space.shape[0]                       # 14 sensors
    ctx_enriched = [hsp["tank1_level"], hsp["tank2_level"], hsp["tank3_level"],
                    *tsp, t_cold, t_amb]                         # 8 context (EnrichedObs)
    h_sp_all = [hsp["tank1_level"], hsp["tank2_level"], hsp["tank3_level"]]

    def numpy_obs():
        levels = [float(obs[0]), float(obs[2]), float(obs[4])]
        temps = [float(obs[1]), float(obs[3]), float(obs[5])]
        return np.array(levels + temps + list(tsp) + h_sp_all + [t_cold, t_amb], dtype=np.float32)

    for k in range(args.steps):
        if expected_shape == (n_raw + len(ctx_enriched),):      # EnrichedObs (train_rl modbus)
            model_obs = np.concatenate([obs, np.array(ctx_enriched, dtype=np.float32)])
        elif expected_shape == (n_raw,):                        # raw 14-dim bridge obs
            model_obs = np.asarray(obs, dtype=np.float32)
        else:                                                   # numpy-track 14-dim obs
            model_obs = numpy_obs()
        action, _ = model.predict(model_obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float64).flatten(), 0.0, 1.0)

        if action_mode == "setpoint":
            # numpy setpoint policy: action = [t_sp0,t_sp1,t_sp2, h_sp0,h_sp1,h_sp2].
            for i, name in enumerate(("tank1_level_sp", "tank2_level_sp", "tank3_level_sp")):
                b.write_register(name, 0.15 + action[3 + i] * 0.30)
            b.write_register("tank1_temp_sp", 20.0 + action[0] * 60.0)
            time.sleep(args.control_dt)
            raw_vars = b.read_raw()
            obs = env._decode_obs(raw_vars)
            sidx = {n: i for i, n in enumerate(env.sensor_names)}
            levels = {n: float(obs[sidx[n]]) for n in env.setpoints}
            temps = {n: float(obs[sidx[n]]) for n in env.temp_setpoints}
            track_l = sum((levels[n] - env.setpoints[n]) ** 2 for n in env.setpoints)
            track_t = sum((temps[n] - env.temp_setpoints[n]) ** 2 for n in env.temp_setpoints)
            w = env.reward_weights
            reward = float(-(w["level"] * track_l + w["temp"] * track_t))
            info = {"levels_m": levels, "temps_c": temps, "raw": raw_vars}
        else:
            obs, reward, _, _, info = env.step(action)           # contract-order 5-MV; env writes directly
            raw_vars = info["raw"]
        rewards.append(reward)
        applied = [float(raw_vars.get(n, 0.0)) / env._act_max[n] for n in env.actuator_names]
        steps_data.append({
            "step": k, "levels": [float(obs[0]), float(obs[2]), float(obs[4])],
            "temps": [float(obs[1]), float(obs[3]), float(obs[5])],
            "action": [float(x) for x in action], "applied_duty": applied,
            "reward": reward, "interlock": detect_interlock(raw_vars)})
        if k % 4 == 0 or k == args.steps - 1:
            lv, tp = info["levels_m"], info.get("temps_c", {})
            LOG.info("step %3d  act=%s  levels(m)=%.3f/%.3f/%.3f  temps(C)=%.1f/%.1f/%.1f  r=%.3f",
                     k, [round(float(x), 2) for x in action],
                     lv.get("tank1_level", float("nan")), lv.get("tank2_level", float("nan")),
                     lv.get("tank3_level", float("nan")), tp.get("tank1_temp", float("nan")),
                     tp.get("tank2_temp", float("nan")), tp.get("tank3_temp", float("nan")), reward)

    env.close()
    LOG.info("rollout done — mean reward = %.4f over %d steps", float(np.mean(rewards)), args.steps)
    model_tag = Path(args.policy).stem
    report(steps_data, tag=f"rl_{model_tag}")


if __name__ == "__main__":
    main()
