#!/usr/bin/env python3
"""Smoke test: on/off interlock-test solenoids SV-1..3 (cabinet only).

The SVs are test instrumentation parallel to V-12/V-23/V-33 and are NOT part of
the agent action space — this exercises their dedicated path instead: the DO
coils (slave 07, FC05/FC01) plus the parallel-bypass physics. Asserts that with
all modulating valves closed, (1) opening SV-1 produces FT-101 flow and moves
level from Tank1 to Tank2, (2) closing it stops the flow, and (3) the SV coils
are absent from the env action space. Uses the bridge's ModbusBackend so the
multi-slave coil dispatch is exercised identically to the test harness.
Requires mock_cabinet.py. Run via ./tests/run_smoke.sh.
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
    scale = next(r["scale"] for r in regs if r["name"] == "init_h1")

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

    def snap():
        raw = be.read_raw()
        return (raw["tank1_level"], raw["tank2_level"], raw["tank1_flow"], raw)

    fails = []
    # (3) SVs are outside the action space
    action_regs = {r["name"] for r in regs if r.get("group") == "actuators"}
    if any(n in action_regs for n in cfg.get("test_actuators", [])):
        fails.append("test_actuators leaked into the actuator (action-space) group")

    reset_to(0.40, 0.10, 0.10)
    # the cabinet is shared across smokes — zero every modulating actuator so
    # this test sees ONLY the SV bypass path (smoke_env's last action otherwise
    # leaves V-12 partly open and FT-101 never settles to 0)
    for n in ("vfd_cmd", "v_12_cmd", "v_23_cmd", "v_33_cmd", "e_101_cmd"):
        be.write_register(n, 0)
    time.sleep(0.5)
    h1_0, h2_0, ft_0, raw0 = snap()
    if abs(ft_0) > 1e-3:
        fails.append(f"baseline FT-101 = {ft_0:.3f} L/min with all valves+SVs closed")

    # (1) open SV-1 (parallel to V-12): Tank1 -> Tank2 full-bore
    be.write_register("sv_1_cmd", True)
    time.sleep(3.0)
    h1_a, h2_a, ft_a, raw_a = snap()
    transfer = h1_0 - h1_a          # mm of level that left Tank1...
    imbalance = transfer - (h2_a - h2_0)   # ...vs what arrived in Tank2

    # (2) close it: flow stops
    be.write_register("sv_1_cmd", False)
    time.sleep(1.0)
    _, _, ft_b, _ = snap()
    be.close()

    print(f"  SV-1 closed: FT-101={ft_0:.2f} L/min  h={h1_0:.3f}/{h2_0:.3f}")
    print(f"  SV-1 open 3s: FT-101={ft_a:.2f} L/min  h={h1_a:.3f}/{h2_a:.3f}  "
          f"(Tank1 -{1000*(h1_0-h1_a):.0f} mm, Tank2 +{1000*(h2_a-h2_0):.0f} mm)")
    print(f"  SV-1 closed again: FT-101={ft_b:.2f} L/min")
    print(f"  coil readback during test: sv_1_cmd={raw_a.get('sv_1_cmd')}")

    if ft_a < 5.0:
        fails.append(f"SV-1 open moved only {ft_a:.2f} L/min on FT-101 (expected >=5)")
    if transfer < 0.02:
        fails.append(f"SV-1 open moved only {1000*transfer:.1f} mm of level in 3s (expected >=20)")
    if abs(imbalance) > 0.004:
        fails.append(f"T1 loss != T2 gain ({1000*transfer:.1f} vs {1000*(h2_a-h2_0):.1f} mm) "
                     f"— bypass path not conserving mass")
    if ft_b > 0.5:
        fails.append(f"FT-101 = {ft_b:.2f} L/min after closing SV-1 (expected ~0)")

    if fails:
        print(f"\n{RED}{BOLD}FAIL{RESET}: " + "; ".join(fails))
        return 1
    print(f"\n{GREEN}{BOLD}PASS{RESET}: SV-1 bypasses V-12 (flow + level shift on "
          f"command, stops on close), and SVs stay out of the action space.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
