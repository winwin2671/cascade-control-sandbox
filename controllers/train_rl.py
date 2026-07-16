"""RL training-track demo — vectorized + time-scaled (G3/G5).

Spins up N cabinets (--time-scale k) and trains SAC/PPO (or runs a random
throughput check). SAC/PPO use SB3's DummyVecEnv over CascadeBridgeEnv factories
(each connecting to a cabinet); the random check uses gymnasium AsyncVectorEnv.

Usage:
    python3 controllers/train_rl.py                                          # random throughput, N=4, k=10
    python3 controllers/train_rl.py --algo ppo --total-timesteps 50000 --device cpu
    python3 controllers/train_rl.py --algo sac --total-timesteps 50000 --device cuda
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
from aio_bridge_env import CascadeBridgeEnv  # noqa: E402

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


def run_sb3(algo: str, pool, n_envs: int, base_port: int, time_scale: float,
            plant_dt: float, total_timesteps: int, device: str = "auto") -> int:
    """Train SAC or PPO on the Modbus track using SB3's own vec env (B1 fix).

    SB3 cannot accept gymnasium AsyncVectorEnv — it needs its own VecEnv.
    We create a DummyVecEnv over CascadeBridgeEnv factories, each connecting
    to a cabinet the pool already spawned."""
    try:
        from stable_baselines3 import SAC, PPO
    except ImportError:
        print("ERROR: stable_baselines3 not installed.\n"
              "  pip3 install --user stable_baselines3   (pulls torch; heavy)")
        return 1
    from stable_baselines3.common.vec_env import DummyVecEnv
    from gymnasium.wrappers import TimeLimit

    wall_dt = plant_dt / time_scale if time_scale > 0 else plant_dt
    # C1 fix: TimeLimit so training episodes truncate + auto-reset (the G1
    # init-state randomization is exercised every episode, not just once per run)
    venv = DummyVecEnv([lambda i=i: TimeLimit(
        CascadeBridgeEnv(backend="modbus", port=base_port + i, control_dt=wall_dt),
        max_episode_steps=200) for i in range(n_envs)])

    if algo == "sac":
        model = SAC("MlpPolicy", venv, device=device, verbose=1, learning_starts=2000,
                    train_freq=1, gradient_steps=4, batch_size=512,
                    policy_kwargs=dict(net_arch=[256, 256]))
    else:
        model = PPO("MlpPolicy", venv, device=device, verbose=1, n_steps=512, batch_size=2048,
                    policy_kwargs=dict(net_arch=[256, 256]))

    model.learn(total_timesteps=total_timesteps)
    out = str(ROOT / "controllers" / f"{algo}_cascade")
    model.save(out)
    import json
    with open(out + ".json", "w") as f:
        json.dump({"action_mode": "actuator", "reward_mode": "track",
                   "algo": algo, "track": "modbus"}, f)
    print(f"{algo.upper()} trained for {total_timesteps} steps; saved to {out}.zip")
    print("(validate it in the IA2 track: load the policy + run it via the RL mode.)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="RL training-track demo (vectorized + time-scaled).")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--time-scale", type=float, default=10.0)
    ap.add_argument("--plant-dt", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=200, help="vec-steps (random policy)")
    ap.add_argument("--algo", default="random", choices=["random", "ppo", "sac"])
    ap.add_argument("--total-timesteps", type=int, default=20000, help="PPO total timesteps")
    ap.add_argument("--device", default="auto", help="cuda | cpu | mps (default: auto)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.algo in ("ppo", "sac"):
        # B1 fix: SB3 needs its own vec env, not gymnasium's AsyncVectorEnv.
        from aio_vec_env import CabinetPool
        print(f"==> starting {args.n_envs} cabinets at {args.time_scale}x (for SB3 {args.algo.upper()})...")
        pool = CabinetPool(args.n_envs, args.time_scale).start()
        try:
            rc = run_sb3(args.algo, pool, args.n_envs, 5020, args.time_scale,
                         args.plant_dt, args.total_timesteps, device=args.device)
        finally:
            pool.close()
    else:
        print(f"==> starting {args.n_envs} cabinets at {args.time_scale}x...")
        env, pool = make_vec_env(n=args.n_envs, time_scale=args.time_scale, plant_dt=args.plant_dt)
        try:
            rc = run_random(env, args.n_envs, args.steps, args.plant_dt)
        finally:
            env.close()
            pool.close()
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
