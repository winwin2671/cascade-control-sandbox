"""Benchmark — compare Manual/PID/MPC(/NMPC) on our 3-tank plant, AIO-Gym style.

Runs each controller through AIOGymNativeEnv('threetank') (our numpy plant, the
kpi/economic/track reward modes + the KPIScorer) and prints a KPI table — the
composite score plus the sub-KPIs (temp/level tracking, excess energy, safety),
meaned over episodes. This is the terminal version of the AIO-Gym-web yardstick.

Usage:
    python3 controllers/benchmark.py                       # Manual/PID/MPC, kpi mode
    python3 controllers/benchmark.py --reward-mode economic --nmpc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from controllers.aiogym_register import register_threetank  # noqa: E402
register_threetank()

import numpy as np  # noqa: E402
from aiogym.env import AIOGymNativeEnv  # noqa: E402
from aiogym.baselines import PIDAgent, make_meas  # noqa: E402
from controllers.mpc_agent import MPCAgent  # noqa: E402


class FixedAgent:
    """Manual baseline: a constant actuator output (no control)."""
    name = "Manual"

    def __init__(self, model, value=0.5):
        nP, nV, nH = model.actuator_counts()
        self.act = {"pumps": [value] * nP, "valves": [value] * nV, "heaters": [value] * nH}

    def reset(self):
        pass

    def compute(self, meas, sp, dt):
        return {k: list(v) for k, v in self.act.items()}


def run(agent, env, episodes, seed):
    """Mirror of aiogym.baselines.evaluate, but also collect per-episode scorer.report()."""
    scores, returns = [], []
    sub = {k: [] for k in ("avg_temp_err", "avg_level_err_cm", "excess_kwh", "interlock_frac")}
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        agent.reset()
        R, done = 0.0, False
        while not done:
            act = agent.compute(make_meas(env), {"h_sp": env.h_sp, "t_sp": env.t_sp}, env.control_dt)
            a = np.array(list(act["pumps"]) + list(act["valves"]) + list(act["heaters"]), dtype=np.float32)
            _, r, term, trunc, _ = env.step(a)
            R += r
            done = term or trunc
        rep = env.scorer.report()
        scores.append(rep["score"])
        returns.append(R)
        for k in sub:
            sub[k].append(rep[k])
    out = {"name": agent.name, "kpi": float(np.mean(scores)), "kpi_std": float(np.std(scores)),
           "return": float(np.mean(returns))}
    for k, v in sub.items():
        out[k] = float(np.mean(v))
    return out


class RLAgent:
    """Trained RL policy (SB3 SAC/PPO) wrapped as an agent for evaluate().

    Direct policies read the env's NATIVE observation (env._obs()) so they see exactly
    what they trained on — including the ∫(sp−meas)dt terms when trained with
    integral_obs (20-D) or the plain 14-D obs otherwise.

    Residual policies (2-D action) get the SAME ResidualEnvWrapper used in training:
    compute() builds the wrapper's 23-D obs (17-D base + integrals), predicts the 2-D residual, and resolves
    it to the full 5-D physical action (feedforward + PID + residual) before returning."""
    def __init__(self, plant, model, env, wrapper=None):
        self.plant = plant                       # for actuator_counts() + controlled_levels()
        self.model = model
        self.env = env                           # native obs source (matches training exactly)
        self.wrapper = wrapper                   # ResidualEnvWrapper or None (direct policy)
        self.name = f"RL-{type(self.model).__name__}" + ("-res" if wrapper else "")

    def reset(self):
        if self.wrapper is not None:
            # benchmark resets the raw env itself; re-sync the wrapper's episodic state
            # (mirrors ResidualEnvWrapper.reset without re-resetting the env)
            from controllers.residual_rl import tracking_steady_state_action
            self.wrapper._reset_episodic_state()
            self.wrapper._model_optimal = tracking_steady_state_action(
                self.wrapper.model, self.wrapper.reference)
            # Minor/#6: the first obs after reset must NOT accumulate the integral
            # (wrapper.reset() passes accumulate=False; training never integrates
            # the reset state). Direct-driving _make_obs with its default would
            # add one extra ∫err step the policy never saw in training.
            self._first_obs = True

    def compute(self, meas, sp, dt):
        if self.wrapper is not None:             # residual paradigm: 23-D obs (17 base + 6 integral), 2-D action
            acc = not getattr(self, "_first_obs", False)
            self._first_obs = False
            obs = self.wrapper._make_obs(self.env._obs(), accumulate=acc)
            action, _ = self.model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
            action = np.clip(np.asarray(action, dtype=np.float64).flatten(), -1.0, 1.0)
            physical = self.wrapper._resolve_action(action)   # canonical [pump,v12,v23,v33,heater]
            # keep the wrapper's prev-action obs in sync — wrapper.step() does this in
            # training/validation; skipping it froze prev_action at 0 and shifted the
            # policy's obs distribution (h3 overfilled, runaway ~89% of steps)
            self.wrapper._prev_physical = physical
            nP, nV, nH = self.plant.actuator_counts()
            return {"pumps": list(physical[:nP]),
                    "valves": list(physical[nP:nP + nV]),
                    "heaters": list(physical[nP + nV:])}
        obs = np.asarray(self.env._obs(), dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float64).flatten(), 0.0, 1.0)
        nP, nV, nH = self.plant.actuator_counts()
        return {"pumps": list(action[:nP]),
                "valves": list(action[nP:nP + nV]),
                "heaters": list(action[nP + nV:])}


def main():
    ap = argparse.ArgumentParser(description="AIO-Gym-style benchmark on the 3-tank plant.")
    ap.add_argument("--reward-mode", default="kpi", choices=["kpi", "economic", "track"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--episode-steps", type=int, default=200)
    ap.add_argument("--nmpc", action="store_true", help="include the NMPC oracle (slow; needs casadi)")
    ap.add_argument("--rl", default=None, help="path to a trained SB3 .zip policy to include")
    args = ap.parse_args()

    env = AIOGymNativeEnv("threetank", reward_mode=args.reward_mode,
                          action_mode="actuator", episode_steps=args.episode_steps,
                          randomize_setpoints=False,   # M5/#6: fixed-reference rows
                          dynamic=False)               # must not be scored against
                                  # setpoints they cannot see. dynamic=False too
                                  # (M5 re-review): the env default is True and its
                                  # disturbance kind 3 is a mid-episode SETPOINT
                                  # MOVE (h_sp/t_sp mutated in place) — 12/20 episodes
                                  # moved the scored setpoints while the residual
                                  # wrapper's reference stayed frozen.
    pairs = [(FixedAgent(env.model), env), (PIDAgent(env.model), env), (MPCAgent(env.model), env)]
    if args.nmpc:
        from controllers.nmpc_oracle import OracleAgent
        pairs.append((OracleAgent(), env))
    if args.rl:
        # Load the policy once and configure the eval env to match how it was trained:
        # obs shape 20 -> integral_obs was on; 14 -> off. RLAgent then reads the env's
        # native obs, so it always matches the training observation exactly.
        from stable_baselines3 import SAC, PPO
        sidecar = args.rl.replace(".zip", ".json")
        meta = json.load(open(sidecar)) if Path(sidecar).exists() else {}
        algo = meta.get("algo", "sac")
        try:
            rl_model = (SAC if algo == "sac" else PPO).load(args.rl)
        except Exception:
            rl_model = PPO.load(args.rl)
        # Modbus-trained DIRECT policies (EnrichedObs, 22-D) can't be benchmarked
        # here: their obs includes bridge-only sensors (3 flows + 5 DI flags) the
        # numpy env doesn't produce — reconstructing them would synthesize sensor
        # values and silently skew the row. Evaluate those on the bridge instead
        # (run_rl.py / validate_policy.py both reconstruct EnrichedObs there).
        is_modbus_direct = (meta.get("action_mode", "actuator") != "residual"
                            and rl_model.observation_space.shape == (22,))
        if is_modbus_direct:
            print(f"ERROR: {args.rl} is a modbus-track direct policy (EnrichedObs 22-D — "
                  "includes flow/DI sensors absent from the numpy env). Benchmark a "
                  "numpy-trained policy, or evaluate this one on the bridge: "
                  f"python3 controllers/run_rl.py --policy {args.rl} --backend ia2")
            sys.exit(1)
        # M6/#6: setpoint-mode policies output SUPERVISORY setpoints, not duties —
        # this benchmark applies the raw action as actuator fractions, which turns
        # temperature setpoints into pump/valve duties and silently slices the 6th
        # dim. (train_sb3's default is setpoint, so this is easy to hit.)
        if meta.get("action_mode") == "setpoint":
            print(f"ERROR: {args.rl} is a setpoint-mode policy (6-D supervisory output). "
                  "The benchmark evaluates direct duties only — retrain with "
                  "--action-mode actuator (or --residual), or validate this policy on "
                  f"the bridge: python3 controllers/run_rl.py --policy {args.rl} "
                  "--action-mode setpoint --backend ia2")
            sys.exit(1)
        rl_env = AIOGymNativeEnv("threetank", reward_mode=args.reward_mode,
                                 action_mode="actuator", episode_steps=args.episode_steps,
                                 randomize_setpoints=False,          # M5/#6: match the base env
                                 dynamic=False,                      # M5 re-review: ditto
                                 integral_obs=(rl_model.observation_space.shape == (20,)))
        # residual policy (2-D action): attach the SAME wrapper used in training
        # (numpy env -> grouped obs indices + canonical action order) so the 2-D
        # residual expands to the full 5-D physical action before env.step().
        # Obs shape 23 = integral_obs on (current), 17 = legacy — match the policy.
        wrapper = None
        if meta.get("action_mode") == "residual" or rl_model.observation_space.shape in ((17,), (23,)):
            from controllers.residual_rl import ResidualEnvWrapper
            from controllers.threetank_model import ThreeTankModel
            _m = ThreeTankModel()
            _hsp, _tsp = _m.default_setpoints()
            _ref = [_hsp[0], _hsp[1], _hsp[2], _tsp[0], _tsp[1], _tsp[2]]
            wrapper = ResidualEnvWrapper(rl_env, _m, _ref,
                                         level_idx=(0, 1, 2), temp_idx=(3, 4, 5),
                                         act_idx=(0, 1, 2, 3, 4),
                                         integral_obs=(rl_model.observation_space.shape == (23,)))
        pairs.append((RLAgent(env.model, rl_model, rl_env, wrapper=wrapper), rl_env))

    results = sorted((run(a, e, args.episodes, 0) for a, e in pairs),
                     key=lambda r: r["kpi"], reverse=True)
    print(f"\n=== Benchmark (mode={args.reward_mode}, {args.episodes} eps x {args.episode_steps} steps) ===")
    hdr = f"{'controller':<10} {'kpi':>7} {'±std':>6} {'temp_err':>8} {'lvl_cm':>7} {'excess_kwh':>10} {'interlock':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<10} {r['kpi']:>7.2f} {r['kpi_std']:>6.2f} {r['avg_temp_err']:>8.2f} "
              f"{r['avg_level_err_cm']:>7.2f} {r['excess_kwh']:>10.3f} {r['interlock_frac']:>9.2f}")


if __name__ == "__main__":
    main()
