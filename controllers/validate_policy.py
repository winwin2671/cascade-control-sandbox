"""Validation gate — validate a trained RL policy on the IA2 track (sim-to-real).

Loads a trained SB3 policy (.zip), runs it through the IA2-in-the-loop track
(CascadeBridgeEnv backend=ia2 — real 50 ms scan + iomap + the L5 safety shield),
computes the same KPI the benchmark uses (via KPIScorer), and reports the sim-to-
real gap.

Handles THREE action modes:
  --action-mode actuator (default): policy outputs the 5 MVs in contract order
      [v_12, v_23, e_101, v_33, vfd] -> env.step writes them directly (PLC mode=mpc).
      numpy-track policies are trained in [pump, V-12, V-23, V-33, heater] order and
      get remapped; modbus-track policies already speak contract order.
  --action-mode setpoint: policy outputs supervisory setpoints (numpy-track setpoint
      policy) -> writes the PID *_sp vars (PLC mode=pid), PID tracks. Requires a
      PLC backend (ia2/edge).
  --action-mode residual: policy outputs a 2D residual [-1,1] (V-33 + heater). The
      env is wrapped with ResidualEnvWrapper so the residual expands to the full
      5D physical action (feedforward + inline PID + residual) before the bridge.

The obs the policy expects depends on the training track; we reconstruct it from
the 14-dim bridge obs + config setpoints, including the integral-of-error terms
for 20-D direct / 23-D residual policies (mirrors run_rl.py).

Requires the IA2 chain up: mock_cabinet.py + ia2-server + cs project open + cs run.

Usage:
    python3 controllers/validate_policy.py --policy controllers/policies/sac_threetank_numpy.zip --backend ia2
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
    ap.add_argument("--action-mode", default=None, choices=["actuator", "setpoint", "residual"],
                    help="Override the policy action mode (otherwise read from the .json "
                         "sidecar written by the trainers). Required when no sidecar is present.")
    args = ap.parse_args()

    from stable_baselines3 import SAC, PPO  # noqa: E402
    from aio_bridge_env import CascadeBridgeEnv  # noqa: E402

    cls = SAC if args.algo == "sac" else PPO
    model = cls.load(args.policy)
    LOG.info("loaded policy: %s (%s)", args.policy, args.algo)

    # Resolve action mode: --action-mode flag > metadata sidecar > hard error.
    # (sidecar also carries the training track — numpy policies output canonical
    # action order, modbus policies contract order; run_rl-style remap below.)
    meta_path = args.policy.replace(".zip", ".json")
    meta = json.load(open(meta_path)) if Path(meta_path).exists() else {}
    if args.action_mode is not None:
        action_mode = args.action_mode
        track = meta.get("track", "numpy")
    elif "action_mode" in meta:
        action_mode = meta["action_mode"]
        track = meta.get("track", "numpy")
        if action_mode not in ("actuator", "setpoint", "residual"):
            LOG.error("Metadata sidecar %s has action_mode=%r; must be 'actuator', "
                      "'setpoint', or 'residual'. Re-train or pass --action-mode.",
                      meta_path, action_mode)
            sys.exit(1)
    else:
        LOG.error("Cannot determine action mode: no sidecar at %s and no --action-mode given.",
                  meta_path)
        sys.exit(1)
    LOG.info("action_mode: %s  track: %s", action_mode, track)

    plant = ThreeTankModel()
    scorer = KPIScorer(plant)
    h_sp_dict, t_sp = plant.default_setpoints()
    h_sp_all = [h_sp_dict.get(i, 0.0) for i in range(plant.n)]
    t_cold, t_amb = plant.t_supply, plant.t_ambient

    plc_mode = "pid" if action_mode == "setpoint" else ("rl" if action_mode == "residual" else "mpc")
    bridge = CascadeBridgeEnv(backend=args.backend, control_dt=args.control_dt, mode=plc_mode)
    b = bridge.backend
    # setpoint mode writes PLC-internal *_sp vars -> needs a PLC backend (ia2/edge).
    if action_mode == "setpoint" and not b.writes_via_plc:
        LOG.error("setpoint-mode policy requires a PLC backend (ia2/edge); the %s backend has "
                  "no *_sp registers. Use --backend ia2, or validate an actuator-mode policy.",
                  args.backend)
        sys.exit(1)
    env = bridge
    if action_mode == "residual":
        # wrap so the 2D residual expands to the full 5D physical action before the
        # bridge. bridge action order [V-12, V-23, E-101, V-33, VFD]; obs shape 23
        # = integral terms on, 17 = legacy policy.
        from controllers.residual_rl import ResidualEnvWrapper
        env = ResidualEnvWrapper(bridge, plant,
                                 [h_sp_dict[0], h_sp_dict[1], h_sp_dict[2], *t_sp],
                                 act_idx=(4, 0, 1, 3, 2),
                                 integral_obs=(model.observation_space.shape[0] == 23))

    obs, _ = env.reset()
    expected_shape = model.observation_space.shape
    n_raw = bridge.observation_space.shape[0]                    # 14 sensors
    am = bridge._act_max                                         # per-actuator eng max (->fraction)
    # context for the EnrichedObs (modbus-track) policy: 3 level sp + 3 temp sp + t_cold + t_amb
    ctx_enriched = [h_sp_dict[i] for i in range(plant.n)] + list(t_sp) + [t_cold, t_amb]
    # integral-of-error terms for 20-D direct policies (numpy obs + 6 integrals —
    # mirrors aiogym's _accumulate_integral; wrapper policies build their own).
    use_integral = (action_mode != "residual" and expected_shape[0] - n_raw == 6)
    itemp, ilevel = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    _I_TEMP_MAX, _I_LEVEL_MAX = 300.0, 8.0

    # numpy-track setpoint/actuator obs reconstruction: [levels(3), temps(3), t_sp(3), h_sp(3), t_cold, t_amb]
    def numpy_obs():
        levels = [float(obs[0]), float(obs[2]), float(obs[4])]
        temps = [float(obs[1]), float(obs[3]), float(obs[5])]
        base = levels + temps + list(t_sp) + h_sp_all + [t_cold, t_amb]
        if use_integral:
            base = base + [v / _I_TEMP_MAX for v in itemp] + [v / _I_LEVEL_MAX for v in ilevel]
        return np.array(base, dtype=np.float32)

    scorer.reset()
    rewards = []
    for k in range(args.steps):
        # build the obs the policy expects
        if action_mode == "residual":
            full_obs = np.asarray(obs, dtype=np.float32)       # wrapper's policy obs (23D/17D)
        elif expected_shape == (n_raw + len(ctx_enriched),):      # EnrichedObs (train_rl modbus)
            full_obs = np.concatenate([obs, np.array(ctx_enriched, dtype=np.float32)])
        elif expected_shape == (n_raw,):                        # raw 14-dim bridge obs
            full_obs = np.asarray(obs, dtype=np.float32)
        else:                                                   # numpy-track obs (14/20-D reconstruction)
            full_obs = numpy_obs()
        action, _ = model.predict(full_obs, deterministic=True)
        lo, hi = (-1.0, 1.0) if action_mode == "residual" else (0.0, 1.0)
        action = np.clip(np.asarray(action, dtype=np.float32).flatten(), lo, hi)

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
            # actuator or residual. numpy actuator policies output TRAINING order
            # [pump, V-12, V-23, V-33, heater]; the bridge's order is CONTRACT
            # [V-12, V-23, E-101, V-33, VFD] — remap or commands land on the wrong
            # actuators. (modbus policies already speak contract order; residual
            # is mapped inside ResidualEnvWrapper._to_env_action.)
            if action_mode == "actuator" and track == "numpy":
                env_action = np.array([action[1], action[2], action[4], action[3], action[0]])
            else:
                env_action = action
            obs, reward, _, _, info = env.step(env_action)
        rewards.append(reward)

        # records from info — obs layout differs per mode (wrapper 23-D vs bridge 14-D)
        lv_d, tp_d = info["levels_m"], info.get("temps_c", {})
        levels = [lv_d["tank1_level"], lv_d["tank2_level"], lv_d["tank3_level"]]
        temps = [tp_d.get("tank1_temp", 0.0), tp_d.get("tank2_temp", 0.0), tp_d.get("tank3_temp", 0.0)]
        if use_integral:                                          # accumulate ∫(sp−meas)dt for next obs
            itemp = [float(np.clip(itemp[i] + (t_sp[i] - temps[i]) * args.control_dt,
                                   -_I_TEMP_MAX, _I_TEMP_MAX)) for i in range(3)]
            ilevel = [float(np.clip(ilevel[j] + (h_sp_all[i] - levels[i]) * args.control_dt,
                                    -_I_LEVEL_MAX, _I_LEVEL_MAX)) for j, i in enumerate((0, 1, 2))]
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
