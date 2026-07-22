"""RL supervisor — drives the live plant with a trained SAC/PPO policy.

Like run_mpc.py / run_nmpc.py but the controller is a trained RL policy. Loads
the .zip, runs it through the IA2 track or directly via Modbus, reports
levels/temps/reward per step.

Handles BOTH action modes:
  --action-mode actuator: policy outputs [p1,p2,h1,h2,h3] -> writes actuator*_req
                           (PLC mode=3 RL, direct). Matches old training default.
  --action-mode setpoint:  policy outputs 5 setpoints (3 temp + 2 level, SUPERVISORY
                           order) -> writes *_sp vars with range mapping (PLC mode=1
                           PID, supervisory RL-on-PID). Matches AIO-Gym's default.

Usage:
    python3 controllers/run_rl.py --policy controllers/sac_threetank.zip --action-mode setpoint --backend ia2
    python3 controllers/run_rl.py --action-mode actuator --backend modbus --policy controllers/sac_cascade.zip

Requires either:
    1) IA2 chain up: mock_cabinet.py + ia2-server + `cs project open` + `cs run`
    2) Modbus track: mock_cabinet.py (direct control, no PLC logic)

Note on edge backend: If using `--backend edge:<name>`, be aware that each
step requires an SSH round-trip proxied through the dev server (~6 handshakes
per 0.5 s step). For edge deployments, increase `--control-dt` to accommodate
the network latency.
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
    ap.add_argument("--policy", default=str(ROOT / "controllers" / "sac_threetank.zip"))
    ap.add_argument("--backend", default="ia2",
                    help="Communication backend: auto | ia2 | modbus | edge:<name> (default: ia2)")
    ap.add_argument("--action-mode", default="setpoint", choices=["actuator", "setpoint"])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--control-dt", type=float, default=0.5)
    args = ap.parse_args()

    # B2 fix: auto-detect action mode from the metadata sidecar (saved by train_sb3.py)
    meta_path = args.policy.replace(".zip", ".json")
    if Path(meta_path).exists():
        meta = json.load(open(meta_path))
        if "action_mode" in meta:
            args.action_mode = meta["action_mode"]

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("pymodbus").setLevel(logging.WARNING)

    from stable_baselines3 import SAC, PPO
    try:
        model = SAC.load(args.policy)
    except Exception:
        model = PPO.load(args.policy)
    LOG.info("loaded policy: %s  action_mode: %s  backend: %s", args.policy, args.action_mode, args.backend)

    cfg = json.load(open(ROOT / "ia2_config.json"))
    hsp = cfg["control"]["setpoints_m"]
    tsp = list(cfg["control"]["setpoints_c"].values())
    t_cold = float(cfg["process"]["t_supply_c"])
    t_amb = float(cfg["process"]["t_ambient_c"])

    # Build the env. For actuator mode, use mode="rl" (writes actuator*_req).
    # For setpoint mode, use mode="pid" (PLC PID active) but write *_sp directly
    # (bypassing env.step's write, which uses the wrong range mapping).
    env = CascadeBridgeEnv(backend=args.backend, control_dt=args.control_dt,
                            mode=("rl" if args.action_mode == "actuator" else "pid"))
    b = env.backend

    obs, _ = env.reset()
    LOG.info("RL supervisor start — action_mode=%s", args.action_mode)
    rewards = []
    steps_data = []

    # Determine expected observation dimensionality from the loaded model
    expected_shape = model.observation_space.shape

    for k in range(args.steps):
        if expected_shape == (13,):
            # Original AIO-Gym 13-dim obs
            levels_list = [float(obs[0]), float(obs[2]), float(obs[4])]
            temps_list = [float(obs[1]), float(obs[3]), float(obs[5])]
            model_obs = np.array(
                levels_list + temps_list + tsp + [hsp["tank1_level"], hsp["tank3_level"]] + [t_cold, t_amb],
                dtype=np.float32)
        else:
            # Native 6-dim obs (used by direct Modbus/Cascade training)
            model_obs = np.asarray(obs, dtype=np.float32)

        action, _ = model.predict(model_obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float64).flatten(), 0.0, 1.0)

        if args.action_mode == "actuator":
            # action = [p1, p2, h1, h2, h3] -> env.step writes actuator*_req.
            obs, reward, _, _, info = env.step(action)
            raw_vars = info["raw"]  # R1 fix: use stashed read (no second read_raw)
        else:
            # action = [t_sp0, t_sp1, t_sp2, h_sp0, h_sp2] (SUPERVISORY order).
            # Map to engineering ranges + write *_sp vars directly.
            t0 = 20.0 + action[0] * 60.0   # 20-80 degC
            t1 = 20.0 + action[1] * 60.0
            t2 = 20.0 + action[2] * 60.0
            h0 = 0.15 + action[3] * 0.40   # 0.15-0.55 m
            h2 = 0.15 + action[4] * 0.40
            b.write_register("tank1_temp_sp", int(round(t0 / 0.01)))
            b.write_register("tank2_temp_sp", int(round(t1 / 0.01)))
            b.write_register("tank3_temp_sp", int(round(t2 / 0.01)))
            b.write_register("tank1_level_sp", int(round(h0 / 0.0001)))
            b.write_register("tank3_level_sp", int(round(h2 / 0.0001)))
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
            info = {"levels_m": levels, "temps_c": temps}

        rewards.append(reward)
        steps_data.append({
            "step": k, "levels": [float(obs[0]), float(obs[2]), float(obs[4])],
            "temps": [float(obs[1]), float(obs[3]), float(obs[5])],
            "action": [float(x) for x in action], "reward": reward,
            "interlock": detect_interlock(raw_vars)})
        if k % 4 == 0 or k == args.steps - 1:
            lv = info.get("levels_m", {}); tp = info.get("temps_c", {})
            LOG.info("step %3d  act=%s  levels(m)=%.3f/%.3f/%.3f  temps(C)=%.1f/%.1f/%.1f  r=%.3f",
                     k, [round(float(x), 2) for x in action],
                     lv.get("tank1_level", float("nan")),
                     lv.get("tank2_level", float("nan")),
                     lv.get("tank3_level", float("nan")),
                     tp.get("tank1_temp", float("nan")),
                     tp.get("tank2_temp", float("nan")),
                     tp.get("tank3_temp", float("nan")), reward)

    env.close()
    LOG.info("rollout done — mean reward = %.4f over %d steps", np.mean(rewards), args.steps)
    model_tag = Path(args.policy).stem  # e.g. sac_threetank, ppo_cascade
    report(steps_data, tag=f"rl_{model_tag}")


if __name__ == "__main__":
    main()