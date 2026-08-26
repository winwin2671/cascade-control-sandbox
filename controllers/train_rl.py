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
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aio_vec_env import make_vec_env  # noqa: E402
from aio_bridge_env import CascadeBridgeEnv  # noqa: E402

LOG = logging.getLogger("train_rl")


def _make_enriched_env(port, control_dt, hsp, tsp, t_cold, t_amb, episode_steps=4000):
    """Wrap CascadeBridgeEnv to append setpoint + ambient context to the raw obs.

    The raw bridge obs is 14-dim (3 levels + 3 temps + 3 flows + 5 DI). We append
    the 3 level setpoints, 3 temp setpoints, and the supply/ambient temperatures
    (8 values) so the policy sees its targets and the heat-sink temperatures."""
    import gymnasium as gym
    from gymnasium.wrappers import TimeLimit

    base = CascadeBridgeEnv(backend="modbus", port=port, control_dt=control_dt)
    n_base = base.observation_space.shape[0]
    ctx = [hsp["tank1_level"], hsp["tank2_level"], hsp["tank3_level"],
           *tsp, t_cold, t_amb]

    class EnrichedObs(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            from gymnasium import spaces
            n = n_base + len(ctx)
            self.observation_space = spaces.Box(
                np.full(n, -np.inf, dtype=np.float32),
                np.full(n, np.inf, dtype=np.float32), dtype=np.float32)

        def observation(self, obs):
            return np.concatenate([obs, np.array(ctx, dtype=np.float32)])

    return TimeLimit(EnrichedObs(base), max_episode_steps=episode_steps)


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
            plant_dt: float, total_timesteps: int, device: str = "auto",
            residual: bool = False, episode_steps: int = 4000) -> int:
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
    from stable_baselines3.common.vec_env import SubprocVecEnv
    import json
    cfg = json.load(open(ROOT / "ia2_config.json"))
    hsp = cfg["control"]["setpoints_m"]
    tsp = list(cfg["control"]["setpoints_c"].values())
    t_cold = float(cfg["process"]["t_supply_c"])
    t_amb = float(cfg["process"]["t_ambient_c"])

    wall_dt = plant_dt / time_scale if time_scale > 0 else plant_dt
    # C1+C5 fix: TimeLimit + EnrichedObs (13-dim, matches AIO-Gym obs) so the
    # policy sees setpoints + t_cold/t_amb, not just the raw 6-dim sensors.
    # R3 fix: SubprocVecEnv (not DummyVecEnv) — each env steps in its own process,
    # so the per-step sleeps overlap (restoring the 0.5 s control period + N× throughput).
    if residual:
        from controllers.residual_rl import ResidualEnvWrapper
        from controllers.threetank_model import ThreeTankModel
        from gymnasium.wrappers import TimeLimit
        _model = ThreeTankModel()
        _ref = [hsp["tank1_level"], hsp["tank2_level"], hsp["tank3_level"], *tsp]
        # bridge obs is interleaved (contract sensor order) → default level/temp idx OK.
        # bridge action order is [V-12, V-23, E-101, V-33, VFD]; canonical physical is
        # [pump, V-12, V-23, V-33, heater], so map each canonical slot to its env slot.
        venv = SubprocVecEnv([lambda i=i: TimeLimit(
            ResidualEnvWrapper(
                CascadeBridgeEnv(backend="modbus", port=base_port + i, control_dt=wall_dt),
                _model, _ref, act_idx=(4, 0, 1, 3, 2)), max_episode_steps=episode_steps)
            for i in range(n_envs)])
    else:
        venv = SubprocVecEnv([lambda i=i: _make_enriched_env(
            base_port + i, wall_dt, hsp, tsp, t_cold, t_amb, episode_steps)
            for i in range(n_envs)])

    if algo == "sac":
        model = SAC("MlpPolicy", venv, device=device, verbose=1, learning_starts=2000,
                    train_freq=1, gradient_steps=4, batch_size=512,
                    policy_kwargs=dict(net_arch=[256, 256]))
    else:
        model = PPO("MlpPolicy", venv, device=device, verbose=1, n_steps=512, batch_size=2048,
                    policy_kwargs=dict(net_arch=[256, 256]))

    model.learn(total_timesteps=total_timesteps)
    # suffix carries the track so modbus policies never collide with the numpy
    # ones (train_sb3.py writes _threetank_numpy / _residual_numpy)
    suffix = "residual_modbus" if residual else "threetank_modbus"
    out = str(ROOT / "controllers" / "policies" / f"{algo}_{suffix}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    import json
    with open(out + ".json", "w") as f:
        json.dump({"action_mode": "residual" if residual else "actuator",
                   "reward_mode": "track", "algo": algo, "track": "modbus"}, f)
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
    ap.add_argument("--episode-steps", type=int, default=4000,
                    help="steps per episode (4000 = 2000 s plant time; the thermal loop "
                         "needs >=4000 for the temperature term to be learnable — at 200 "
                         "steps temp rises ~2 C and the policy learns nothing about heat)")
    ap.add_argument("--device", default="auto", help="cuda | cpu | mps (default: auto)")
    ap.add_argument("--residual", action="store_true",
                    help="use xinji's residual RL (2D action: V-33 + heater on top of model feedforward)")
    args = ap.parse_args()

    # footgun guard: --algo defaults to "random" (throughput check). Someone
    # passing --residual / --episode-steps almost certainly meant to train.
    if args.algo == "random" and (args.residual or args.episode_steps != ap.get_default("episode_steps")):
        ap.error("--residual/--episode-steps require --algo sac|ppo (default algo is 'random', "
                 "which only runs the throughput check)")

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.algo in ("ppo", "sac"):
        # B1 fix: SB3 needs its own vec env, not gymnasium's AsyncVectorEnv.
        from aio_vec_env import CabinetPool
        print(f"==> starting {args.n_envs} cabinets at {args.time_scale}x (for SB3 {args.algo.upper()})...")
        pool = CabinetPool(args.n_envs, args.time_scale).start()
        try:
            rc = run_sb3(args.algo, pool, args.n_envs, 5200, args.time_scale,
                         args.plant_dt, args.total_timesteps, device=args.device,
                         residual=args.residual, episode_steps=args.episode_steps)
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
