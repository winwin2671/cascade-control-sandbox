#!/usr/bin/env python3
"""Disturbance sidecar: random valve-fault injection through SV-1..3.

Runs each interlock-test solenoid as an independent random telegraph —
random start delay -> OPEN -> random hold -> CLOSED -> random hold -> repeat
— to simulate on/off valve faults while a controller runs, so the control
model's fault response can be observed (levels drift, FT-10x see the bypass
flow, LSH/LSL trips become reachable). Designed to run in the background
next to run_mode.sh's foreground controller (`--disturbance`), but is a
plain standalone script: point --host/--port at a real cabinet or --server
at a real IA2/edge runtime for hardware testing.

Write paths (pick with --backend; run_mode.sh derives it from the mode):
  ia2    PLC owns the coils — threetank.st drives `sv_*_cmd := sv_*_req AND
         test_sv_en` every 50 ms scan, so raw FC05 writes get reclaimed. The
         sanctioned path is the PLC-internal `sv_*_req` BOOLs + `test_sv_en`
         gate via the IA2 variable API (same as env.set_test_valve).
  modbus no PLC in the loop — write the slave-07 coils directly (FC05).

The intended valve state is re-asserted on a heartbeat (default 0.5 s), so
CascadeBridgeEnv.reset() force-closing the SVs at episode start cannot
swallow a disturbance: the hold resumes within one heartbeat. Physics is
frozen while reset_cmd is asserted, so the re-assert never skews init levels.

Every run writes a JSONL log (header with seed + params, one record per
transition, exit footer) — the only record that a rollout saw disturbances.
Replay a run exactly with `--seed N` plus the same --valves and hold ranges
(all captured in the header record). NOTE: numpy+gymnasium are imported via
aio_bridge_env (hard deps of this repo); on a hardware-only host without
them, the two backend classes would need extracting first.

Cleanup contract: SIGINT/SIGTERM closes every SV and clears test_sv_en
before exit. kill -9 skips that — coils stay in their last state until the
next boot (every run_mode.sh run reboots cabinet + PLC). Concurrent sidecars
are unsupported: two schedules fight over the same coils.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from aio_bridge_env import IA2Backend, ModbusBackend, load_config

ROOT = Path(__file__).resolve().parent
VALVES = ("sv_1", "sv_2", "sv_3")
DEFAULT_LOG_DIR = ROOT / "controllers" / "runs"
VERIFY_BUDGET_S = 1.5      # startup read-back verify: writes must stick within this
MAX_CONSEC_FAILURES = 3    # failed writes in a row -> give up loudly


# --------------------------------------------------------------------------- #
# Schedule core — pure, shared by the runtime loop and the smoke-test oracle.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TelegraphParams:
    seed: int
    valves: tuple[str, ...]                # participating valves, canonical order
    start_min: float
    start_max: float
    open_min: float                        # hold range while OPEN (fault duration)
    open_max: float
    closed_min: float                      # hold range while CLOSED (recovery time)
    closed_max: float


@dataclass(frozen=True)
class Transition:
    t: float          # scheduled time (s since arm; the write lands <=0.1 s later)
    valve: str
    opened: bool
    next_hold: float  # hold drawn for the state this transition entered


class TelegraphPlan:
    """N independent random telegraphs over one random.Random(seed).

    Draws are taken lazily in a fixed sequence — earliest deadline first,
    ties broken by valve index — so wall-clock jitter changes WHEN draws
    happen but never their order: `--seed` replay is exact. Both the runtime
    loop (poll with a monotonic clock) and simulate() (virtual clock stepping
    event-to-event) drive the same _fire(), consuming identical draws.
    """

    def __init__(self, params: TelegraphParams):
        self.params = params
        self.rng = random.Random(params.seed)
        self.opened = {v: False for v in params.valves}
        self.deadline = {v: self._draw(params.start_min, params.start_max)
                         for v in params.valves}  # start delays, valve order

    def _draw(self, lo: float, hi: float) -> float:
        return self.rng.uniform(lo, hi)

    def _fire(self, valve: str) -> Transition:
        """Flip `valve` at its (due) deadline and draw the next hold."""
        p = self.params
        t = self.deadline[valve]
        self.opened[valve] = not self.opened[valve]
        lo, hi = (p.open_min, p.open_max) if self.opened[valve] else (p.closed_min, p.closed_max)
        hold = self._draw(lo, hi)
        self.deadline[valve] = t + hold
        return Transition(t=t, valve=valve, opened=self.opened[valve], next_hold=hold)

    def poll(self, now: float) -> list[Transition]:
        """Fire every valve whose deadline is due, in canonical valve order."""
        return [self._fire(v) for v in self.params.valves if self.deadline[v] <= now]

    def next_deadline(self) -> float:
        return min(self.deadline.values())


def simulate(params: TelegraphParams, duration: float) -> list[Transition]:
    """Materialize the schedule up to `duration` — the replay oracle.

    Walks a virtual clock event-to-event (same earliest-deadline/index rule
    as the runtime loop), so its transitions are bit-identical to what a run
    with the same seed/params/valves produces.
    """
    plan = TelegraphPlan(params)
    out: list[Transition] = []
    while plan.next_deadline() < duration:
        out.extend(plan.poll(plan.next_deadline()))
    return out


# --------------------------------------------------------------------------- #
# Backend adapter — env.set_test_valve semantics against a bare backend.
# --------------------------------------------------------------------------- #
class DisturbanceWriter:
    """One valve-write API over both tracks.

    IA2 track: writes the PLC-internal `sv_*_req` vars; the POU ANDs them
    with `test_sv_en` before driving the DO coils. Modbus track: writes the
    slave-07 coils directly (no gate — the coils are the state).
    """

    def __init__(self, backend):
        self.backend = backend
        self.via_plc = bool(backend.writes_via_plc)

    def write_valve(self, valve: str, opened: bool) -> None:
        name = f"{valve}_req" if self.via_plc else f"{valve}_cmd"
        self.backend.write_register(name, bool(opened))

    def set_enabled(self, enabled: bool) -> None:
        if self.via_plc:
            self.backend.write_register("test_sv_en", bool(enabled))

    def read_valves(self) -> dict:
        """Suffix-keyed raw dict (`sv_*_req` on ia2, `sv_*_cmd` on modbus)."""
        return self.backend.read_raw()

    def close(self) -> None:
        self.backend.close()


def build_writer(args: argparse.Namespace, cfg: dict) -> DisturbanceWriter:
    target = (f"IA2 server {args.server}" if args.backend == "ia2"
              else f"cabinet {args.host}:{args.port}")
    try:
        if args.backend == "ia2":
            return DisturbanceWriter(IA2Backend(args.server, args.project))
        return DisturbanceWriter(ModbusBackend(args.host, args.port, cfg["registers"]))
    except Exception as e:
        raise RuntimeError(
            f"cannot reach {target}: {e}\n"
            f"  hint: is mock_cabinet running? For --backend ia2: is ia2-server "
            f"up and `cs run` done?") from None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_valves(spec: str) -> tuple[str, ...]:
    out = set()
    for tok in spec.split(","):
        tok = tok.strip().lower().removeprefix("sv_")
        if not tok.isdigit() or not (1 <= int(tok) <= len(VALVES)):
            raise argparse.ArgumentTypeError(
                f"invalid valve '{tok.strip()}' — use 1..{len(VALVES)} (e.g. --valves 1,3)")
        out.add(f"sv_{tok}")
    if not out:
        raise argparse.ArgumentTypeError("--valves selected nothing")
    return tuple(v for v in VALVES if v in out)   # canonical order


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Random SV-1..3 valve-fault disturbance sidecar "
                    "(see run_mode.sh --disturbance).")
    p.add_argument("--backend", choices=("ia2", "modbus"), default="modbus",
                   help="ia2: write sv_*_req/test_sv_en via the IA2 variable API "
                        "(PLC owns the coils); modbus: write slave-07 FC05 coils "
                        "directly. Default %(default)s.")
    p.add_argument("--host", default=None,
                   help="cabinet host (modbus). Default $CABINET_HOST or config.")
    p.add_argument("--port", type=int, default=None,
                   help="cabinet port (modbus). Default $CABINET_PORT or config.")
    p.add_argument("--server", default=None,
                   help="IA2 server URL (ia2). Default $IA2_SERVER or config.")
    p.add_argument("--project", default=None,
                   help="IA2 project name for the X-IA2-Project header. Default config.")
    p.add_argument("--config", default=None,
                   help="path to ia2_config.json (default: next to this script).")
    p.add_argument("--valves", type=parse_valves, default=VALVES,
                   help="participating valves, e.g. 1,3 or sv_1,sv_3. Default all.")
    p.add_argument("--seed", type=int, default=None,
                   help="schedule seed (default: random, printed + logged for replay).")
    p.add_argument("--start-min", type=float, default=3.0,
                   help="min start delay per valve, s (default %(default)s).")
    p.add_argument("--start-max", type=float, default=8.0)
    p.add_argument("--open-min", type=float, default=2.0,
                   help="min OPEN hold, s (default %(default)s).")
    p.add_argument("--open-max", type=float, default=6.0)
    p.add_argument("--closed-min", type=float, default=10.0,
                   help="min CLOSED hold, s (default %(default)s).")
    p.add_argument("--closed-max", type=float, default=25.0)
    p.add_argument("--heartbeat", type=float, default=0.5,
                   help="intended-state re-assert period, s (default %(default)s). "
                        "Survives env.reset() closing the SVs at episode start.")
    p.add_argument("--duration", type=float, default=None,
                   help="stop injecting after this many s (default: until signaled).")
    p.add_argument("--log", default=None,
                   help="JSONL log path (default: controllers/runs/"
                        "disturbance_sidecar_YYYYMMDD_HHMMSS.jsonl).")
    p.add_argument("--verbose", action="store_true",
                   help="also log heartbeat re-asserts.")
    args = p.parse_args(argv)

    for lo, hi, what in ((args.start_min, args.start_max, "start"),
                         (args.open_min, args.open_max, "open"),
                         (args.closed_min, args.closed_max, "closed")):
        if not (0.0 < lo <= hi):
            p.error(f"--{what}-min/--{what}-max must satisfy 0 < min <= max "
                    f"(got {lo}/{hi})")
    if args.heartbeat <= 0.0:
        p.error("--heartbeat must be > 0")
    if args.duration is not None and args.duration <= 0.0:
        p.error("--duration must be > 0")
    return args


def resolve_targets(args: argparse.Namespace, cfg: dict) -> None:
    """Fill host/port/server/project defaults: flag > env > ia2_config.json."""
    m, ia2 = cfg["modbus"], cfg["ia2"]
    args.host = args.host or os.environ.get("CABINET_HOST") or m["host"]
    args.port = int(args.port or os.environ.get("CABINET_PORT") or m["port"])
    args.server = args.server or os.environ.get("IA2_SERVER") or ia2["server_url"]
    args.project = args.project if args.project is not None else ia2["project_name"]


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
class Stopper:
    def __init__(self):
        self.event = threading.Event()
        self.reason = "duration"

    def arm_signals(self) -> None:
        def _handler(signum, _frame):
            self.reason = signal.Signals(signum).name.lower()
            self.event.set()   # no I/O in a signal handler
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)


def verify_closed(writer: DisturbanceWriter, params: TelegraphParams) -> None:
    """Confirm the arm writes reached the plant (read-back, with retry).

    On ia2 the POU needs one 50 ms scan for sv_*_req/test_sv_en to show up,
    and two reads landing inside one scan trip IA2Backend's frozen-scan
    (C7) guard — both are retries, not failures; real failure is the budget
    expiring.
    """
    suffix = "_req" if writer.via_plc else "_cmd"
    end = time.monotonic() + VERIFY_BUDGET_S
    while time.monotonic() < end:
        try:
            raw = writer.read_valves()
            closed = all(not raw.get(f"{v}{suffix}", False) for v in params.valves)
            gated_ok = (not writer.via_plc) or bool(raw.get("test_sv_en"))
            if closed and gated_ok:
                return
        except RuntimeError as e:
            if "frozen" not in str(e).lower():
                raise
        time.sleep(0.1)
    raise RuntimeError(
        "SV writes are not reaching the plant (read-back did not settle within "
        f"{VERIFY_BUDGET_S}s) — check the iomap/device wiring "
        "(ia2) or the DO slave (modbus).")


def run(argv: list[str]) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    resolve_targets(args, cfg)

    seed = args.seed if args.seed is not None else secrets.randbits(31)
    params = TelegraphParams(
        seed=seed, valves=tuple(args.valves),
        start_min=args.start_min, start_max=args.start_max,
        open_min=args.open_min, open_max=args.open_max,
        closed_min=args.closed_min, closed_max=args.closed_max)

    log_path = Path(args.log) if args.log else DEFAULT_LOG_DIR / (
        "disturbance_sidecar_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"==> [disturbance] seed={seed} backend={args.backend} "
          f"valves={','.join(params.valves)} "
          f"start={args.start_min:.0f}-{args.start_max:.0f}s "
          f"open={args.open_min:.0f}-{args.open_max:.0f}s "
          f"closed={args.closed_min:.0f}-{args.closed_max:.0f}s "
          f"(replay: --seed {seed} + same valves/ranges)", flush=True)
    print(f"==> [disturbance] log: {log_path}", flush=True)

    try:
        writer = build_writer(args, cfg)
    except RuntimeError as e:
        print(f"[disturbance] ERROR: {e}", file=sys.stderr, flush=True)
        return 3

    exit_code = 0
    stop = Stopper()
    stop.arm_signals()
    t0_epoch: float = 0.0
    try:
        writer.read_valves()          # liveness probe (ia2: surfaces "no program running")
        writer.set_enabled(True)      # arm: open the gate, close every SV
        for v in VALVES:
            writer.write_valve(v, False)
        verify_closed(writer, params)

        intended = {v: False for v in VALVES}   # non-participants stay closed
        plan = TelegraphPlan(params)
        counts = {"transitions": 0, "heartbeats": 0, "write_failures": 0}
        failures = 0

        def guarded(fn, *a) -> None:
            """Count consecutive write failures; 3 in a row aborts the run."""
            nonlocal failures
            try:
                fn(*a)
                failures = 0
            except Exception as e:
                failures += 1
                counts["write_failures"] += 1
                if failures >= MAX_CONSEC_FAILURES:
                    raise RuntimeError(
                        f"{MAX_CONSEC_FAILURES} consecutive SV writes failed "
                        f"(last: {e})") from None
                print(f"[disturbance] write failed ({failures}/"
                      f"{MAX_CONSEC_FAILURES}): {e}", flush=True)

        t0_epoch = time.time()
        t0 = time.monotonic()
        with open(log_path, "w", buffering=1) as jl:
            def rec(obj: dict) -> None:
                jl.write(json.dumps(obj) + "\n")

            rec({"type": "header", "seed": seed,
                 "params": {k: v for k, v in asdict(params).items() if k != "seed"},
                 "backend": args.backend,
                 "server": args.server if writer.via_plc else None,
                 "host": None if writer.via_plc else f"{args.host}:{args.port}",
                 "heartbeat_s": args.heartbeat,
                 "t0_epoch": t0_epoch, "pid": os.getpid(), "argv": sys.argv})

            next_hb = 0.0
            rel = 0.0
            while not stop.event.is_set():
                rel = time.monotonic() - t0
                if args.duration is not None and rel >= args.duration:
                    break
                for tr in plan.poll(rel):
                    guarded(writer.write_valve, tr.valve, tr.opened)
                    intended[tr.valve] = tr.opened
                    counts["transitions"] += 1
                    state = "OPEN" if tr.opened else "CLOSED"
                    rec({"type": "event", "t": round(tr.t, 3),
                         "epoch": t0_epoch + tr.t, "valve": tr.valve,
                         "state": state.lower(), "next_hold_s": round(tr.next_hold, 3)})
                    print(f"[disturbance] t=+{tr.t:.2f}s "
                          f"{tr.valve.replace('sv_', 'SV-').upper()} {state} "
                          f"(hold {tr.next_hold:.2f}s)", flush=True)
                if rel >= next_hb:      # idempotent re-assert (survives env.reset)
                    for v, opened in intended.items():
                        guarded(writer.write_valve, v, opened)
                    guarded(writer.set_enabled, True)   # ia2 only; survives POU restart
                    counts["heartbeats"] += 1
                    next_hb = rel + args.heartbeat   # re-arm from now: no burst after a stall
                    if args.verbose:
                        rec({"type": "heartbeat", "t": round(rel, 3),
                             "state": {v: intended[v] for v in params.valves}})
                wake = min(t0 + plan.next_deadline(), t0 + next_hb,
                            time.monotonic() + 0.1)   # cap: never skip a deadline
                time.sleep(max(0.0, wake - time.monotonic()))

            rec({"type": "exit", "reason": stop.reason, "uptime_s": round(rel, 3),
                 "counts": counts})
    except RuntimeError as e:
        print(f"[disturbance] ERROR: {e}", file=sys.stderr, flush=True)
        exit_code = 3
        stop.reason = "error"
        try:  # best-effort error record if the log was already open
            with open(log_path, "a", buffering=1) as jl:
                jl.write(json.dumps({"type": "exit", "reason": "error",
                                     "error": str(e), "epoch": time.time()}) + "\n")
        except OSError:
            pass
    finally:
        for v in VALVES:               # every close independently guarded — one
            try:                       # dead link must not abort the rest
                writer.write_valve(v, False)
            except Exception:
                pass
        try:
            writer.set_enabled(False)
        except Exception:
            pass
        writer.close()
        print(f"==> [disturbance] exit ({stop.reason}); SVs closed, gate cleared",
              flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
