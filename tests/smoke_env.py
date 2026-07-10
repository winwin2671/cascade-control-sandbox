#!/usr/bin/env python3
"""Smoke test: the Gym env (reset/step/reward) over the Modbus backend.

Builds CascadeBridgeEnv in modbus mode (cabinet direct — no IA2), resets a few
times (checking init levels are randomized AND applied), and steps (checking obs
+ reward are sane). Requires mock_cabinet.py. Run via ./tests/run_smoke.sh.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aio_bridge_env import CascadeBridgeEnv  # noqa: E402

GREEN, RED, BOLD, RESET = "\033[32m", "\033[31m", "\033[1m", "\033[0m"


def main() -> int:
    try:
        env = CascadeBridgeEnv(backend="modbus", control_dt=0.3)
    except Exception as e:
        print(f"{RED}FAIL{RESET}: cannot build env (modbus backend): {e}\n"
              f"  start mock_cabinet.py first (or run ./tests/run_smoke.sh).")
        return 1

    fails = []
    seen_init = []
    for ep in range(3):
        obs, info = env.reset(seed=ep + 1)
        init = [round(v, 3) for v in info.get("init_levels_m", {}).values()]
        obs_levels = [round(float(obs[i]), 3) for i in (0, 2, 4)]  # h1, h2, h3
        seen_init.append(tuple(init))
        # cabinet holds init levels while reset_cmd is asserted, so obs == init
        if not all(abs(a - b) <= 0.02 for a, b in zip(obs_levels, init)):
            fails.append(f"ep{ep}: obs levels {obs_levels} != sampled init {init}")
        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(action)
        act_str = [round(float(a), 2) for a in action]
        print(f"  ep{ep}: init={init}  act={act_str}  reward={reward:.3f}")
        if reward >= 0:
            fails.append(f"ep{ep}: reward {reward:.3f} should be < 0 (off setpoint)")
    env.close()

    if len(set(seen_init)) < 2:
        fails.append("init levels not randomized across episodes")
    if fails:
        print(f"\n{RED}{BOLD}FAIL{RESET}: " + "; ".join(fails))
        return 1
    print(f"\n{GREEN}{BOLD}PASS{RESET}: env reset (randomized), step, and reward work over modbus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
