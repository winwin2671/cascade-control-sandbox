#!/usr/bin/env python3
"""Smoke test: episode reset through the Modbus contract (cabinet only).

Asserts that writing target levels to init_h1/2/3 and pulsing a nonce on
reset_cmd snaps the cabinet's tank levels to those targets — and that two
different targets both take effect (the reset is controllable, not one-shot).

Requires mock_cabinet.py running on the contract host:port. Run via
./tests/run_smoke.sh (boots the cabinet for you) — no IA2 needed.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from pymodbus.client import AsyncModbusTcpClient

ROOT = Path(__file__).resolve().parents[1]
GREEN, RED, BOLD, RESET = "\033[32m", "\033[31m", "\033[1m", "\033[0m"
INIT_REGS = ["init_h1", "init_h2", "init_h3"]   # -> tanks 1, 2, 3
TOL = 0.02  # m


def contract():
    cfg = json.loads((ROOT / "ia2_config.json").read_text())
    return cfg, {r["name"]: r for r in cfg["registers"]}


async def _connect(cl):
    for _ in range(30):
        if await cl.connect():
            return True
        await asyncio.sleep(0.5)
    return False


async def main() -> int:
    cfg, reg = contract()
    m = cfg["modbus"]
    import os
    host = os.environ.get("CABINET_HOST", m["host"])
    port = int(os.environ.get("CABINET_PORT", m["port"]))
    unit = int(os.environ.get("CABINET_UNIT", m["unit_id"]))
    scale = reg["tank1_level"]["scale"]

    cl = AsyncModbusTcpClient(host=host, port=port)
    if not await _connect(cl):
        print(f"{RED}FAIL{RESET}: cannot connect to cabinet at {host}:{port} — "
              f"start mock_cabinet.py first (or run ./tests/run_smoke.sh).")
        return 1

    async def w(name, val):
        r = await cl.write_registers(reg[name]["address"], [int(val)], device_id=unit)
        if r.isError():
            raise RuntimeError(str(r))

    async def levels():
        r = await cl.read_holding_registers(reg["tank1_level"]["address"],
                                            count=6, device_id=unit)
        return [round(r.registers[i] * scale, 3) for i in (0, 2, 4)]  # h1, h2, h3

    nonce = 0

    async def reset_to(targets):
        nonlocal nonce
        nonce += 1
        for name, t in zip(INIT_REGS, targets):
            await w(name, round(t / scale))
        await w("reset_cmd", nonce)          # fresh nonce -> cabinet snaps + holds
        await asyncio.sleep(0.3)
        got = await levels()
        await w("reset_cmd", 0)              # release -> cabinet resumes
        return got

    failures = []
    for targets in ([0.40, 0.20, 0.35], [0.15, 0.45, 0.25]):
        got = await reset_to(targets)
        ok = all(abs(g - t) <= TOL for g, t in zip(got, targets))
        print(f"  reset -> {targets}  got {got}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append((targets, got))
    cl.close()

    if failures:
        print(f"\n{RED}{BOLD}FAIL{RESET}: reset did not snap to targets: {failures}")
        return 1
    print(f"\n{GREEN}{BOLD}PASS{RESET}: reset snaps tank levels to the requested init values.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
