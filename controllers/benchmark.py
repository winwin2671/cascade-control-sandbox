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
    """Trained RL policy (SB3 SAC/PPO) wrapped as an agent for evaluate()."""
    def __init__(self, model_path):
        from stable_baselines3 import SAC, PPO
        import json as _json
        # Read algo from metadata sidecar; fall back to trying SAC then PPO
        algo = "sac"
        sidecar = model_path.replace(".zip", ".json")
        if Path(sidecar).exists():
            algo = _json.load(open(sidecar)).get("algo", "sac")
        try:
            self.model = (SAC if algo == "sac" else PPO).load(model_path)
        except Exception:
            self.model = PPO.load(model_path)
        self.name = f"RL-{type(self.model).__name__}"

    def reset(self):
        pass

    def compute(self, meas, sp, dt):
        ctrl = [0, 2]   # controlled_levels
        obs = (meas["levels"] + meas["temps"] + list(sp["t_sp"])
               + [sp["h_sp"][i] for i in ctrl] + [meas["t_cold"], meas["t_amb"]])
        action, _ = self.model.predict(np.array(obs, dtype=np.float32), deterministic=True)
        action = np.clip(np.asarray(action, dtype=np.float64).flatten(), 0.0, 1.0)
        return {"pumps": list(action[:2]), "valves": [], "heaters": list(action[2:])}


def main():
    ap = argparse.ArgumentParser(description="AIO-Gym-style benchmark on the 3-tank plant.")
    ap.add_argument("--reward-mode", default="kpi", choices=["kpi", "economic", "track"])
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--episode-steps", type=int, default=200)
    ap.add_argument("--nmpc", action="store_true", help="include the NMPC oracle (slow; needs casadi)")
    ap.add_argument("--rl", default=None, help="path to a trained SB3 .zip policy to include")
    args = ap.parse_args()

    env = AIOGymNativeEnv("threetank", reward_mode=args.reward_mode,
                          action_mode="actuator", episode_steps=args.episode_steps)
    pairs = [(PIDAgent(env.model), env), (MPCAgent(env.model), env)]
    if args.nmpc:
        from controllers.nmpc_oracle import OracleAgent
        pairs.append((OracleAgent(), env))
    if args.rl:
        # RL trained in setpoint mode (picks targets, PID tracks) -> evaluate on
        # a setpoint-mode env so its output is interpreted correctly. PID/MPC
        # stay on the actuator-mode env. The KPI scores are comparable (same
        # plant + scorer + reward mode).
        rl_env = AIOGymNativeEnv("threetank", reward_mode=args.reward_mode,
                                  action_mode="setpoint", episode_steps=args.episode_steps)
        pairs.append((RLAgent(args.rl), rl_env))

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
