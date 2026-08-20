#!/usr/bin/env python3
"""Shield regression test — the ST safety layer under automated test (#6 H6 offer).

The smoke suite deliberately bypasses IA2, so every threetank.st edit has been
manually verified only (yichi's H6) — the E-stop polarity bug (B1) sat in exactly
that blind spot. This test closes it: it boots the REAL chain (mock_cabinet +
ia2-server + the ThreeTank POU via cs) and drives trip conditions through the
physical path (actuator commands -> physics -> sensors/DI -> shield -> mapped
outputs), asserting the shield's response each time:

  S0  healthy passthrough   — no latch with all DIs healthy (B1 regression: the
                              NC e-stop reads 1 and nothing trips)
  S1  overflow              — pump cut (inflow stopped), drains FORCED open above
                              0.40 despite a shut command, autonomous recovery
  S2  dry-fire              — heater cut, pump NOT cut (the refill path survives)
  S3  e-stop (forced DI)    — pressing (di_estop=0 via cs force) latches all
                              three trips; releasing recovers

Exit code 1 on any assertion failure. Slow-ish (~2-3 min): it runs the real PLC
scan loop against real physics (4x time-scale).

Usage: python3 tests/shield_regression.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GREEN, RED, BOLD, RESET = "\033[32m", "\033[31m", "\033[1m", "\033[0m"
IA2 = ROOT / "ia2" / "target" / "release"
TIME_SCALE = float(os.environ.get("SHIELD_TIME_SCALE", "4"))
CTRL_DT = 0.5 / TIME_SCALE          # wall seconds per step -> 0.5 plant-s per step


def _wait_port(port: int, timeout: float = 15.0) -> bool:
    import socket
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def _wait_health(url: str, timeout: float = 20.0) -> bool:
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.4)
    return False


class Stack:
    """Boot cabinet + ia2-server + POU; tear down on exit."""

    def __init__(self):
        self.procs: list[subprocess.Popen] = []
        self.force_pinned: list[str] = []

    def start(self):
        if not _wait_port(5020, timeout=0.2):     # already up? use it, else boot
            self.procs.append(subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "mock_cabinet.py"),
                 "--time-scale", str(TIME_SCALE), "--log-every", "0"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            if not _wait_port(5020):
                raise RuntimeError("mock_cabinet did not open :5020")
        if not _wait_health("http://127.0.0.1:3001/api/health", timeout=0.5):
            if not (IA2 / "server").exists():
                raise RuntimeError(f"{IA2}/server not built — cargo build --release in ia2/")
            self.procs.append(subprocess.Popen(
                [str(IA2 / "server"), "--bind", "127.0.0.1:3001"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            if not _wait_health("http://127.0.0.1:3001/api/health"):
                raise RuntimeError("ia2-server did not become healthy")
        for cmd in (["project", "open", str(ROOT / "ia2_project")], ["run"]):
            r = subprocess.run([str(IA2 / "cs")] + cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"cs {' '.join(cmd)} failed: {r.stderr.strip()[:200]}")

    def force(self, name: str, value: str):
        r = subprocess.run([str(IA2 / "cs"), "runtime", "force", name, value],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"cs force {name}={value} failed: {r.stderr.strip()[:200]}")
        self.force_pinned.append(name)

    def unforce(self, name: str):
        r = subprocess.run([str(IA2 / "cs"), "runtime", "unforce", name],
                           capture_output=True, text=True)
        if r.returncode == 0 and name in self.force_pinned:
            self.force_pinned.remove(name)

    def stop(self):
        for name in list(self.force_pinned):
            self.unforce(name)
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


def main() -> int:
    from aio_bridge_env import CascadeBridgeEnv
    from controllers.rollout_report import detect_interlock

    fails: list[str] = []
    stack = Stack()
    try:
        stack.start()
        env = CascadeBridgeEnv(backend="ia2", control_dt=CTRL_DT, mode="rl")
        b = env.backend

        def sraw(out) -> dict:
            """Stashed post-step snapshot from env.step's info (R1 pattern) — a
            fresh read_raw() right after a step lands in the same 50 ms scan and
            trips the C7 frozen-obs guard."""
            return out[4]["raw"]

        def u16(rr: dict, name: str) -> int:
            """Field from a CACHED snapshot — read_raw twice inside one 50 ms scan
            trips the C7 frozen-obs guard, so never re-read per field."""
            return int(float(rr.get(name, 0)))

        def step(action, n=1):
            out = None
            for _ in range(n):
                out = env.step(action)
                if out[2] or out[3]:
                    env.reset()
            return out

        # ---- S0: healthy passthrough (B1 regression) ------------------------
        env.reset()
        out = step([0.5, 0.5, 0.4, 0.5, 0.6], n=8)
        r0 = sraw(out)                                 # ONE snapshot, many fields
        pairs = [("vfd_cmd_req", "vfd_cmd"), ("v_12_cmd_req", "v_12_cmd"),
                 ("v_23_cmd_req", "v_23_cmd"), ("v_33_cmd_req", "v_33_cmd"),
                 ("e_101_cmd_req", "e_101_cmd")]
        for req, mapped in pairs:
            # req is REAL percent; mapped is UINT raw (x100, the POU's REAL_TO_INT)
            if abs(u16(r0, mapped) - u16(r0, req) * 100) > 300:   # ~3% + quantization
                fails.append(f"S0: {mapped}={u16(r0, mapped)} raw != req "
                             f"{u16(r0, req)}% with healthy DIs "
                             f"(untripped passthrough broken — B1 class)")
        if detect_interlock(r0):
            fails.append("S0: interlock active with all DIs healthy")
        print(f"  S0 healthy passthrough: mapped≈req, no interlock "
              f"{'OK' if not fails else 'FAIL'}")

        # ---- S1: overflow — pump cut, drains forced, autonomous recovery ----
        env.reset()
        # bridge order [V-12, V-23, E-101, V-33, VFD]: pump full, V-12 nearly shut
        tripped = None
        for k in range(220):
            rr = sraw(env.step([0.03, 0.9, 0.5, 0.9, 1.0]))
            if float(rr.get("tank1_level", 0)) > 0.45:
                tripped = k
                break
            # assert inside the loop with the SAME snapshot
        if tripped is None:
            fails.append("S1: tank1 never crossed 0.45 m in 220 steps (physics?)")
        else:
            if u16(rr, "vfd_cmd") != 0:
                fails.append(f"S1: pump NOT cut on overflow (vfd_cmd={u16(rr, 'vfd_cmd')})")
            if u16(rr, "v_12_cmd") < 9000:
                fails.append(f"S1: drain not forced open (v_12_cmd={u16(rr, 'v_12_cmd')} "
                             f"vs ~300 commanded)")
            recovered, rr2 = None, None
            for k in range(260):
                rr2 = sraw(env.step([0.03, 0.9, 0.5, 0.9, 1.0]))
                if float(rr2.get("tank1_level", 1)) < 0.40:
                    recovered = k
                    break
            if recovered is None:
                fails.append("S1: level never recovered below 0.40 (latch stuck?)")
            elif u16(rr2, "vfd_cmd") < 3000:
                fails.append(f"S1: pump did not resume after recovery (vfd_cmd={u16(rr2, 'vfd_cmd')})")
            print(f"  S1 overflow: pump cut @step {tripped}, drain forced open, "
                  f"recovered {'OK' if not [f for f in fails if f.startswith('S1')] else 'FAIL'}")

        # ---- S2: dry-fire — heater cut, pump alive (fails safe, recoverable) -
        env.reset()
        tripped = None
        for k in range(300):
            rr = sraw(env.step([1.0, 1.0, 0.5, 1.0, 0.5]))   # V-12 full, pump 50%: h1 drains
            if float(rr.get("tank1_level", 1)) < 0.05:
                tripped = k
                break
        if tripped is None:
            fails.append("S2: tank1 never drained below 0.05 m in 300 steps")
        else:
            if u16(rr, "e_101_cmd") != 0:
                fails.append(f"S2: heater NOT cut on dry-fire (e_101_cmd={u16(rr, 'e_101_cmd')})")
            if u16(rr, "vfd_cmd") < 3000:
                fails.append(f"S2: pump cut by dry-fire — refill path dead "
                             f"(vfd_cmd={u16(rr, 'vfd_cmd')})")
            print(f"  S2 dry-fire: heater cut @step {tripped}, pump alive "
                  f"{'OK' if not [f for f in fails if f.startswith('S2')] else 'FAIL'}")

        # ---- S3: e-stop via forced DI (NC: 0 = pressed) ----------------------
        env.reset()
        step([0.5, 0.5, 0.4, 0.5, 0.6], n=4)
        stack.force("di_estop", "0")                  # press
        rp = sraw(step([0.5, 0.5, 0.4, 0.5, 0.6], n=6))
        s3_bad = []
        if u16(rp, "vfd_cmd") != 0 or u16(rp, "e_101_cmd") != 0:
            s3_bad.append(f"pressed: vfd={u16(rp, 'vfd_cmd')} e101={u16(rp, 'e_101_cmd')} "
                          f"(both must be 0)")
        stack.unforce("di_estop")                     # release
        rr3 = sraw(step([0.5, 0.5, 0.4, 0.5, 0.6], n=6))
        if u16(rr3, "vfd_cmd") < 3000 or u16(rr3, "e_101_cmd") < 3000:
            s3_bad.append(f"released: vfd={u16(rr3, 'vfd_cmd')} e101={u16(rr3, 'e_101_cmd')} "
                          f"(must resume — levels are healthy after reset)")
        fails.extend(f"S3: {m}" for m in s3_bad)
        print(f"  S3 e-stop (forced DI): press cuts pump+heater, release recovers "
              f"{'OK' if not s3_bad else 'FAIL'}")

        env.close()
    except Exception as e:
        fails.append(f"harness: {e}")
    finally:
        stack.stop()

    if fails:
        print(f"\n{RED}{BOLD}FAIL{RESET}: " + "; ".join(fails))
        return 1
    print(f"\n{GREEN}{BOLD}PASS{RESET}: shield regression S0-S3 — ST safety layer verified "
          f"through the real PLC scan loop.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
