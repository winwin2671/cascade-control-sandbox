"""RL training (SAC / PPO) on our 3-tank plant, AIO-Gym style.

register_threetank() + AIOGymNativeEnv('threetank', reward_mode, action_mode) in a
SubprocVecEnv (AIO-Gym's train.py pattern), then SB3 SAC or PPO with AIO-Gym's
hyperparams. Outputs a saved policy (.zip). Same algorithm + setup as AIO-Gym's
train.py, but on our plant.

Needs: pip install --user torch (CUDA build) stable_baselines3

Usage:
    python3 controllers/train_sb3.py --algo sac --reward-mode economic --steps 30000
    python3 controllers/train_sb3.py --algo ppo --reward-mode kpi --steps 30000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from controllers.aiogym_register import register_threetank  # noqa: E402
register_threetank()

from aiogym.env import AIOGymNativeEnv  # noqa: E402
from stable_baselines3 import SAC, PPO  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor  # noqa: E402


def make_env(seed, reward_mode, action_mode, episode_steps):
    def _f():
        env = AIOGymNativeEnv("threetank", reward_mode=reward_mode, action_mode=action_mode,
                              episode_steps=episode_steps, randomize_plant=True, dynamic=True)
        env.reset(seed=seed)
        return env
    return _f


def best_device():
    """CUDA if present (real GPU), else CPU. Apple MPS avoided (per-op overhead
    dominates tiny MLPs); the parallel env workers are the real accelerator.
    Override with --device mps if you want to try MPS."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description="RL training (SAC/PPO) on the 3-tank plant.")
    ap.add_argument("--algo", default="sac", choices=["sac", "ppo"])
    ap.add_argument("--reward-mode", default="economic", choices=["kpi", "economic", "track"])
    ap.add_argument("--action-mode", default="setpoint", choices=["actuator", "setpoint"],
                    help="setpoint = RL picks targets, PID tracks (AIO-Gym default; easier)")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--episode-steps", type=int, default=400)
    ap.add_argument("--grad-steps", type=int, default=4, help="SAC update-to-data ratio")
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: auto)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    venv = SubprocVecEnv([make_env(1000 + i, args.reward_mode, args.action_mode, args.episode_steps)
                          for i in range(args.n_envs)])
    venv = VecMonitor(venv)

    device = args.device or best_device()
    print(f"device: {device}  action_mode: {args.action_mode}  reward: {args.reward_mode}")

    if args.algo == "sac":
        model = SAC("MlpPolicy", venv, device=device, verbose=1, learning_starts=2000,
                    train_freq=1, gradient_steps=args.grad_steps, batch_size=512,
                    policy_kwargs=dict(net_arch=[256, 256]))
    else:
        model = PPO("MlpPolicy", venv, device=device, verbose=1, n_steps=512, batch_size=2048,
                    policy_kwargs=dict(net_arch=[256, 256]))

    model.learn(total_timesteps=args.steps, progress_bar=False)

    out = args.out or str(ROOT / "controllers" / f"{args.algo}_threetank")
    model.save(out)
    # save metadata sidecar (B2 fix: lets validate_policy.py + run_rl.py auto-detect
    # the action mode — setpoint vs actuator — instead of guessing wrong)
    import json as _json
    with open(out + ".json", "w") as f:
        _json.dump({"action_mode": args.action_mode, "reward_mode": args.reward_mode,
                     "algo": args.algo}, f)
    print(f"\nsaved {out}.zip  +  {out}.json (action_mode={args.action_mode})")


if __name__ == "__main__":
    main()
