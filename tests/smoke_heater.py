#!/usr/bin/env python3
"""Smoke test: heater actuation + level->temperature coupling (cabinet only).

Asserts that (1) a heater at full duty raises its tank's temperature, and
(2) opening the cold pump inflow *slows* that heating — the cascade disturbance
(the level loop pumping cold supply water into the tank the heater is trying to
warm). Requires mock_cabinet.py. Run via ./tests/run_smoke.sh.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from pymodbus.client import AsyncModbusTcpClient

ROOT = Path(__file__).resolve().parents[1]
GREEN, RED, BOLD, RESET = "\033[32m", "\033[31m", "\033[1m", "\033[0m"


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
    lscale, tscale = reg["tank1_level"]["scale"], reg["tank1_temp"]["scale"]

    cl = AsyncModbusTcpClient(host=host, port=port)
    if not await _connect(cl):
        print(f"{RED}FAIL{RESET}: cannot connect to cabinet at {host}:{port} — "
              f"start mock_cabinet.py first (or run ./tests/run_smoke.sh).")
        return 1

    async def w(name, val):
        r = await cl.write_registers(reg[name]["address"], [int(val)], device_id=unit)
        if r.isError():
            raise RuntimeError(str(r))

    async def t1():
        r = await cl.read_holding_registers(reg["tank1_temp"]["address"],
                                            count=1, device_id=unit)
        return round(r.registers[0] * tscale, 2)

    # reset to a known level (h1=0.30) so the thermal mass is known
    await w("init_h1", round(0.30 / lscale))
    await w("init_h2", round(0.18 / lscale))
    await w("init_h3", round(0.24 / lscale))
    await w("reset_cmd", 1)
    await asyncio.sleep(0.3)
    await w("reset_cmd", 0)
    t0 = await t1()

    await w("heater1", 10000)                # heater1 full duty, pumps off
    await asyncio.sleep(4.0)
    t1a = await t1()

    await w("actuator1", 10000)              # now pump1 full -> cold supply into Tank1
    await asyncio.sleep(4.0)
    t1b = await t1()
    cl.close()

    d_no_pump, d_pump = t1a - t0, t1b - t1a
    print(f"  T1: start={t0}  +heater 4s -> {t1a} (d={d_no_pump:+.1f})  "
          f"+pump 4s -> {t1b} (d={d_pump:+.1f})")

    fails = []
    if d_no_pump < 3.0:
        fails.append(f"heater raised T1 only {d_no_pump:.1f} C in 4s (expected >=3)")
    if d_pump >= d_no_pump:
        fails.append(f"cold pump inflow did not slow heating "
                     f"(d_pump={d_pump:.1f} >= d_no_pump={d_no_pump:.1f})")
    if fails:
        print(f"\n{RED}{BOLD}FAIL{RESET}: " + "; ".join(fails))
        return 1
    print(f"\n{GREEN}{BOLD}PASS{RESET}: heater raises T1 and cold pump inflow slows it "
          f"(the cascade disturbance).")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
