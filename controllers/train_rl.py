"""RL training-track demo — vectorized + time-scaled (G3/G5).

Spins up N cabinets (--time-scale k) + an AsyncVectorEnv and runs a policy,
reporting the throughput (plant-s / wall-s — the combined G3/G5 speedup). The
default policy is random (proves the track works); --algo ppo trains with SB3
PPO if stable_baselines3 is installed.

Usage:
    python3 controllers/train_rl.py                              # random, N=4, k=10
    python3 controllers/train_rl.py --n-envs 8 --time-scale 20 --steps 500
    python3 controllers/train_rl.py --algo ppo --total-timesteps 20000   # needs SB3
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aio_vec_env import make_vec_env  # noqa: E402

LOG = logging.getLogger("train_rl")


def run_random(env, n_envs: int, steps: int, plant_dt: float) -> int:
    obs, _ = env.reset()
    t0 = time.time()
    for _ in range(steps):
        obs, reward, term, trunc, info = env.step(env.action_space.sample())
    wall = time.time() - t0
    plant_steps = steps * n_envs
    plant_s = plant_steps * plant_dt
    print(f"\n{steps} vec-steps x {n_envs} envs = {plant_steps} plant-steps in {wall:.1f}s wall")
    print(f"throughput: {plant_steps / wall:.1f} plant-steps/s  =  "
          f"{plant_s / wall:.1f} plant-s/s  ({plant_s / wall:.0f}x real-time)")
    return 0


def run_ppo(env, total_timesteps: int) -> int:
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("ERROR: stable_baselines3 not installed.\n"
              "  pip3 install --user stable_baselines3   (pulls torch; heavy)")
        return 1
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    out = str(ROOT / "controllers" / "ppo_cascade.zip")
    model.save(out)
    print(f"PPO trained for {total_timesteps} steps; saved to {out}")
    print("(validate it in the IA2 track: load the policy + run it via the RL mode.)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="RL training-track demo (vectorized + time-scaled).")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--time-scale", type=float, default=10.0)
    ap.add_argument("--plant-dt", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=200, help="vec-steps (random policy)")
    ap.add_argument("--algo", default="random", choices=["random", "ppo"])
    ap.add_argument("--total-timesteps", type=int, default=20000, help="PPO total timesteps")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(f"==> starting {args.n_envs} cabinets at {args.time_scale}x...")
    env, pool = make_vec_env(n=args.n_envs, time_scale=args.time_scale, plant_dt=args.plant_dt)
    try:
        rc = run_ppo(env, args.total_timesteps) if args.algo == "ppo" else \
            run_random(env, args.n_envs, args.steps, args.plant_dt)
    finally:
        env.close()
        pool.close()
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
