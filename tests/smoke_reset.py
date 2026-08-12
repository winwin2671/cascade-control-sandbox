#!/usr/bin/env python3
"""Smoke test: episode reset through the multi-slave contract (cabinet only).

Asserts that writing target levels to init_h1/2/3 (slave 03, FC06) and pulsing a
nonce on reset_cmd (slave 03) snaps the cabinet's tank levels to those targets —
and that two different targets both take effect. Levels are read back from slave
02 (FC04 floats). Requires mock_cabinet.py. Run via ./tests/run_smoke.sh.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aio_bridge_env import ModbusBackend, load_config  # noqa: E402

GREEN, RED, BOLD, RESET = "\033[32m", "\033[31m", "\033[1m", "\033[0m"
LEVEL_REGS = ["tank1_level", "tank2_level", "tank3_level"]
INIT_REGS = ["init_h1", "init_h2", "init_h3"]
TOL = 0.02  # m


def main() -> int:
    cfg = load_config()
    m = cfg["modbus"]
    host = os.environ.get("CABINET_HOST", m["host"])
    port = int(os.environ.get("CABINET_PORT", m["port"]))
    regs = cfg["registers"]
    scale = next(r["scale"] for r in regs if r["name"] == "init_h1")

    try:
        be = ModbusBackend(host, port, regs)
    except Exception as e:
        print(f"{RED}FAIL{RESET}: cannot connect to cabinet at {host}:{port} — {e}\n"
              f"  start mock_cabinet.py first (or run ./tests/run_smoke.sh).")
        return 1

    nonce = 0

    def reset_to(targets):
        nonlocal nonce
        nonce += 1
        for name, t in zip(INIT_REGS, targets):
            be.write_register(name, round(t / scale))   # uint16 raw -> slave 03 FC06
        be.write_register("reset_cmd", nonce)            # fresh nonce -> snap + hold
        time.sleep(0.3)
        raw = be.read_raw()
        got = [round(raw[n], 3) for n in LEVEL_REGS]     # floats from slave 02 FC04
        be.write_register("reset_cmd", 0)                # release -> resume
        return got

    failures = []
    for targets in ([0.40, 0.20, 0.35], [0.15, 0.45, 0.25]):
        got = reset_to(targets)
        ok = all(abs(g - t) <= TOL for g, t in zip(got, targets))
        print(f"  reset -> {targets}  got {got}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append((targets, got))
    be.close()

    if failures:
        print(f"\n{RED}{BOLD}FAIL{RESET}: reset did not snap to targets: {failures}")
        return 1
    print(f"\n{GREEN}{BOLD}PASS{RESET}: reset snaps tank levels to the requested init values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
