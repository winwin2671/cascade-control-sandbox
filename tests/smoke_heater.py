#!/usr/bin/env python3
"""Smoke test: heater actuation + recirculation coupling (cabinet only).

Asserts that (1) the E-101 heater at full duty raises Tank1 temperature, and
(2) turning on the recirculation pump (VFD) *slows* that heating — the pump
pushes reservoir-temperature water back into Tank1 (the cascade disturbance).
Uses the bridge's ModbusBackend so the multi-slave / float dispatch is exercised
identically to training. Requires mock_cabinet.py. Run via ./tests/run_smoke.sh.
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


def main() -> int:
    cfg = load_config()
    m = cfg["modbus"]
    host = os.environ.get("CABINET_HOST", m["host"])
    port = int(os.environ.get("CABINET_PORT", m["port"]))
    regs = cfg["registers"]
    scale = next(r["scale"] for r in regs if r["name"] == "init_h1")  # uint16 level raw scale

    try:
        be = ModbusBackend(host, port, regs)
    except Exception as e:
        print(f"{RED}FAIL{RESET}: cannot connect to cabinet at {host}:{port} — {e}\n"
              f"  start mock_cabinet.py first (or run ./tests/run_smoke.sh).")
        return 1

    def reset_to(h1, h2, h3):
        be.write_register("init_h1", round(h1 / scale))
        be.write_register("init_h2", round(h2 / scale))
        be.write_register("init_h3", round(h3 / scale))
        be.write_register("reset_cmd", 1)
        time.sleep(0.3)
        be.write_register("reset_cmd", 0)

    def t1():
        return be.read_raw()["tank1_temp"]

    reset_to(0.30, 0.22, 0.18)
    t0 = t1()

    be.write_register("e_101_cmd", 10000)        # heater full duty (u16 raw 10000 = 100%), pump off
    time.sleep(10.0)
    t1a = t1()

    be.write_register("vfd_cmd", 10000)          # now pump full -> cold recirc into Tank1
    time.sleep(10.0)
    t1b = t1()
    be.write_register("e_101_cmd", 0.0)
    be.write_register("vfd_cmd", 0.0)
    be.close()

    d_no_pump, d_pump = t1a - t0, t1b - t1a
    print(f"  T1: start={t0:.2f}  +heater 10s -> {t1a:.2f} (d={d_no_pump:+.2f})  "
          f"+pump 10s -> {t1b:.2f} (d={d_pump:+.2f})")

    fails = []
    if d_no_pump < 0.10:
        fails.append(f"heater raised T1 only {d_no_pump:.2f} C in 10s (expected >=0.10; 2kW, ~23.5L at 0.30m)")
    if d_no_pump - d_pump < 0.01:
        fails.append(f"recirc coupling too weak (d_no_pump={d_no_pump:.2f} - d_pump={d_pump:.2f} "
                     f"= {d_no_pump - d_pump:.2f} < 0.01; advection coupling may be missing)")
    if fails:
        print(f"\n{RED}{BOLD}FAIL{RESET}: " + "; ".join(fails))
        return 1
    print(f"\n{GREEN}{BOLD}PASS{RESET}: heater raises T1 and recirc pump inflow slows it "
          f"(the cascade disturbance).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
