"""RL training (SAC / PPO) on our 3-tank plant, AIO-Gym style.

register_threetank() + AIOGymNativeEnv('threetank', reward_mode, action_mode) in a
SubprocVecEnv (AIO-Gym's train.py pattern), then SB3 SAC or PPO with AIO-Gym's
hyperparams. Outputs a saved policy (.zip). Same algorithm + setup as AIO-Gym's
train.py, but on our plant.

Needs: pip install --user torch (CUDA build) stable_baselines3

Usage:
    # 5D direct actuator (RL drives all 5 MVs)
    python3 controllers/train_sb3.py --algo sac --action-mode actuator --reward-mode regulation --steps 30000
    # 2D residual (xinji: V-33 + heater on top of feedforward + PID) — same numpy env
    python3 controllers/train_sb3.py --algo sac --residual --steps 30000
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


def make_env(seed, reward_mode, action_mode, episode_steps, residual=False):
    def _f():
        # residual paradigm needs the raw 5D actuator action expanded from the 2D
        # residual, so the underlying env must be in actuator mode regardless of
        # the --action-mode the user passed.
        eff_mode = "actuator" if residual else action_mode
        env = AIOGymNativeEnv("threetank", reward_mode=reward_mode, action_mode=eff_mode,
                              episode_steps=episode_steps, randomize_plant=True, dynamic=True)
        from controllers.threetank_model import ThreeTankModel
        _m = ThreeTankModel()
        _hsp, _tsp = _m.default_setpoints()
        _ref = [_hsp[0], _hsp[1], _hsp[2], _tsp[0], _tsp[1], _tsp[2]]
        # numpy AIOGymNativeEnv obs is GROUPED [levels(3), temps(3), ...] and its
        # action is canonical [pump, V-12, V-23, V-33, heater] — pass those indices.
        if residual:
            from controllers.residual_rl import ResidualEnvWrapper
            env = ResidualEnvWrapper(env, _m, _ref,
                                     level_idx=(0, 1, 2), temp_idx=(3, 4, 5),
                                     act_idx=(0, 1, 2, 3, 4))
        elif reward_mode == "regulation":
            from controllers.residual_rl import RegulationRewardWrapper
            env = RegulationRewardWrapper(env, _m, _ref,
                                          level_idx=(0, 1, 2), temp_idx=(3, 4, 5))
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
    ap.add_argument("--reward-mode", default="economic", choices=["kpi", "economic", "track", "regulation"])
    ap.add_argument("--action-mode", default="setpoint", choices=["actuator", "setpoint"],
                    help="setpoint = RL picks targets, PID tracks (default). "
                         "actuator = RL drives the 5 MVs directly.")
    ap.add_argument("--residual", action="store_true",
                    help="xinji's residual RL: 2D action (V-33 + heater) on top of model "
                         "feedforward + inline PID. Forces actuator mode. Compare against "
                         "5D direct (no --residual) on the SAME numpy env — same physics, "
                         "same step budget, so the gap isolates the paradigm, not the track.")
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--episode-steps", type=int, default=400)
    ap.add_argument("--grad-steps", type=int, default=4, help="SAC update-to-data ratio")
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: auto)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    venv = SubprocVecEnv([make_env(1000 + i, args.reward_mode, args.action_mode,
                                   args.episode_steps, residual=args.residual)
                          for i in range(args.n_envs)])
    venv = VecMonitor(venv)

    device = args.device or best_device()
    paradigm = "residual (2D)" if args.residual else f"direct ({args.action_mode})"
    print(f"device: {device}  paradigm: {paradigm}  reward: {args.reward_mode}")

    if args.algo == "sac":
        model = SAC("MlpPolicy", venv, device=device, verbose=1, learning_starts=2000,
                    train_freq=1, gradient_steps=args.grad_steps, batch_size=512,
                    policy_kwargs=dict(net_arch=[256, 256]))
    else:
        model = PPO("MlpPolicy", venv, device=device, verbose=1, n_steps=512, batch_size=2048,
                    policy_kwargs=dict(net_arch=[256, 256]))

    model.learn(total_timesteps=args.steps, progress_bar=False)

    suffix = "residual" if args.residual else "threetank"
    out = args.out or str(ROOT / "controllers" / "policies" / f"{args.algo}_{suffix}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    # save metadata sidecar (B2 fix: lets validate_policy.py + run_rl.py auto-detect
    # the action mode — setpoint vs actuator — instead of guessing wrong)
    import json as _json
    with open(out + ".json", "w") as f:
        _json.dump({"action_mode": "residual" if args.residual else args.action_mode,
                     "reward_mode": args.reward_mode, "algo": args.algo,
                     "track": "numpy"}, f)
    print(f"\nsaved {out}.zip  +  {out}.json "
          f"(paradigm={'residual' if args.residual else args.action_mode})")


if __name__ == "__main__":
    main()
