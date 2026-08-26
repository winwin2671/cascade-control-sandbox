#!/usr/bin/env python3
"""Smoke test: the disturbance sidecar drives the SV coils on schedule.

Cabinet-only (modbus track). Launches disturbance_sidecar.py as a subprocess
with a fixed seed and fast holds, then asserts:
  1. determinism — the JSONL event sequence is identical to the pure
     simulate() oracle for the same seed/params/valves (the replay contract);
  2. command-following — coil read-back matches the schedule at every
     observer sample (guard band around each transition), and sv_2 stays
     closed throughout (--valves subset honored);
  3. the disturbance moves the plant — FT-101 >= 5 L/min while SV-1's coil
     is energized (same threshold smoke_sv proves on this reset state);
  4. cleanup — after the --duration exit every coil reads False.
Requires mock_cabinet.py. Run via ./tests/run_smoke.sh.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aio_bridge_env import ModbusBackend, load_config  # noqa: E402
from disturbance_sidecar import TelegraphParams, simulate  # noqa: E402

GREEN, RED, BOLD, RESET = "\033[32m", "\033[31m", "\033[1m", "\033[0m"

SEED = 7
PARAMS = TelegraphParams(seed=SEED, valves=("sv_1", "sv_3"),
                         start_min=0.2, start_max=0.4,
                         open_min=1.0, open_max=1.2,
                         closed_min=1.0, closed_max=1.2)
DURATION_S = 5.0
GUARD_S = 0.4          # samples this close to a transition are not asserted


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

    # baseline: known levels, modulating actuators zeroed, SVs closed (the
    # cabinet is shared across smokes) — makes the FT-101 threshold robust
    reset_to(0.40, 0.10, 0.10)
    for n in ("vfd_cmd", "v_12_cmd", "v_23_cmd", "v_33_cmd", "e_101_cmd"):
        be.write_register(n, 0)
    for n in cfg["test_actuators"]:
        be.write_register(n, False)
    time.sleep(0.5)

    expected = simulate(PARAMS, DURATION_S)
    if len(expected) < 3:
        print(f"{RED}FAIL{RESET}: degenerate schedule ({len(expected)} events) — "
              f"pick a different seed.")
        return 1

    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "disturbance.jsonl"
        cmd = [sys.executable, "-u", str(ROOT / "disturbance_sidecar.py"),
               "--backend", "modbus", "--host", host, "--port", str(port),
               "--seed", str(SEED), "--valves", "1,3",
               "--start-min", "0.2", "--start-max", "0.4",
               "--open-min", "1.0", "--open-max", "1.2",
               "--closed-min", "1.0", "--closed-max", "1.2",
               "--duration", str(DURATION_S), "--heartbeat", "0.2",
               "--log", str(log)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)

        # observer: sample coil read-back (+ FT-101) while the sidecar runs
        samples = []          # (epoch, sv_1, sv_2, sv_3, ft101)
        obs_end = time.time() + DURATION_S + 4.0
        while time.time() < obs_end:
            raw = be.read_raw()
            samples.append((time.time(), bool(raw["sv_1_cmd"]),
                            bool(raw["sv_2_cmd"]), bool(raw["sv_3_cmd"]),
                            raw["tank1_flow"]))
            time.sleep(0.1)
        out, _ = proc.communicate(timeout=10)
        log_text = log.read_text() if log.exists() else ""   # read before the tempdir dies

    # ---- 1. determinism: JSONL events == simulate() oracle ----
    fails = []
    if not log_text.strip():
        print(f"{RED}FAIL{RESET}: sidecar wrote no JSONL log.\nsidecar output:\n{out}")
        return 1
    records = [json.loads(line) for line in log_text.splitlines()]
    header = records[0]
    events = [r for r in records if r.get("type") == "event"]
    if header.get("seed") != SEED:
        fails.append(f"header seed {header.get('seed')} != {SEED}")
    if tuple(header["params"]["valves"]) != PARAMS.valves:
        fails.append(f"header valves {header['params']['valves']} != {PARAMS.valves}")
    if len(events) != len(expected):
        fails.append(f"{len(events)} JSONL events vs {len(expected)} oracle events")
    else:
        for got, want in zip(events, expected):
            if (got["valve"], got["state"] == "open") != (want.valve, want.opened) \
                    or abs(got["t"] - want.t) > 1e-3:
                fails.append(f"event mismatch: JSONL {got} vs oracle {want}")
                break

    # ---- 2. command-following against the schedule ----
    t0 = header["t0_epoch"]
    boundaries = [tr.t for tr in expected] + [DURATION_S]   # duration end = shutdown edge

    def intended(valve: str, rel: float) -> bool:
        state = False
        for tr in expected:
            if tr.valve == valve and tr.t <= rel:
                state = tr.opened
        return state

    checked = 0
    ft101_open_max = 0.0
    for epoch, sv1, sv2, sv3, ft101 in samples:
        rel = epoch - t0
        if rel >= DURATION_S:
            continue                     # past --duration: sidecar has shut down
        if any(abs(rel - b) < GUARD_S for b in boundaries):
            continue                     # transition in flight — not asserted
        checked += 1
        for coil, valve in ((sv1, "sv_1"), (sv2, "sv_2"), (sv3, "sv_3")):
            want = intended(valve, rel) if valve in PARAMS.valves else False
            if coil != want:
                fails.append(f"t=+{rel:.2f}s {valve} coil={coil} but schedule "
                             f"says {'OPEN' if want else 'CLOSED'}")
                break
        if sv1:
            ft101_open_max = max(ft101_open_max, ft101)

    # ---- 3. the disturbance moves the plant ----
    if ft101_open_max < 5.0:
        fails.append(f"FT-101 peaked at {ft101_open_max:.2f} L/min while SV-1's "
                     f"coil was energized (expected >=5)")

    # ---- 4. cleanup after --duration exit ----
    footer = records[-1]
    if footer.get("type") != "exit" or footer.get("reason") != "duration":
        fails.append(f"footer is {footer} — expected exit/duration")
    if proc.returncode != 0:
        fails.append(f"sidecar exit code {proc.returncode} (expected 0)")
    time.sleep(0.3)
    raw = be.read_raw()
    be.close()
    for n in cfg["test_actuators"]:
        if raw[n]:
            fails.append(f"{n} still energized after sidecar exit")

    print(f"  schedule: {len(expected)} transitions, {checked}/{len(samples)} "
          f"samples asserted (rest inside the ±{GUARD_S}s guard)")
    for tr in expected:
        print(f"    t=+{tr.t:.3f}s {tr.valve.replace('sv_', 'SV-').upper()} "
              f"{'OPEN' if tr.opened else 'CLOSED'} (hold {tr.next_hold:.2f}s)")
    print(f"  FT-101 while SV-1 energized: max {ft101_open_max:.2f} L/min")

    if fails:
        print(f"\nsidecar output:\n{out}")
        print(f"\n{RED}{BOLD}FAIL{RESET}: " + "; ".join(fails[:5]))
        return 1
    print(f"\n{GREEN}{BOLD}PASS{RESET}: sidecar follows its seeded schedule on the "
          f"coils (subset honored), moves the plant, and closes everything on exit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
