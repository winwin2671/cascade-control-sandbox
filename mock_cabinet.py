#!/usr/bin/env python3
"""mock_cabinet.py — three-tank "decoupled" process simulator (Modbus TCP server).

A pymodbus TCP *server* (slave) bound to 127.0.0.1:5020 that exposes the
holding-register map described in `ia2_config.json`.  An asyncio physics loop
updates the sensor registers (levels and temperatures) every tick from
whatever actuator-command registers a Modbus master (the IA2 engine, a
hand-written controller, or an RL agent) writes.

**Config-driven.**  The register layout (names, addresses, directions) and the
engineering<->raw scales are read from `ia2_config.json` — the same single
contract `tools/gen_ia2_artifacts.py` uses to generate the IA2 device/iomap
TOMLs, so the cabinet, the iomap, and the bridge env can never drift apart.
Adding a register (reset, heaters, ...) is a contract edit; this file picks it
up automatically.  What stays *here* is the process semantics: the 3-tank
topology and which register name maps to which physics state.

Process topology — canonical three-tank benchmark (Amira DTS200-style):

    pump q1 --> Tank 1
    pump q2 --> Tank 3
    Tank 1 --(valve a1)--> Tank 2 --(valve a3)--> Tank 3   (hydraulic coupling
                                                          through the middle tank)
    Tank 2 --(valve a2)--> reservoir (drain)

Tank 1 and Tank 3 are each fed by their own pump, but their levels are
hydraulically coupled through Tank 2.  Driving Tank 1 / Tank 3 levels
*independently* in the face of that coupling is the classic *decoupling*
control problem — hence "Tank 3 (Decoupled)".

Register encoding (uint16), defined in ia2_config.json:  engineering = raw * scale.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import signal
from dataclasses import dataclass
from pathlib import Path

from pymodbus.server import ModbusTcpServer
from pymodbus.simulator import SimData, SimDevice
from pymodbus.simulator.simutils import DataType

CONFIG_PATH = Path(__file__).resolve().parent / "ia2_config.json"

# --------------------------------------------------------------------------- #
# Physics model constants (NOT in the contract — these are internal tuning).
# Contract values (q_max / h_max / t_supply) come from ia2_config.json instead.
# --------------------------------------------------------------------------- #
G = 9.81  # gravity, m/s^2
A_TANK = 0.0154  # tank cross-sectional area, m^2
S_PIPE = 5.0e-5  # connecting-pipe cross-section, m^2
A1 = 0.5  # outflow coefficient, Tank1<->Tank2 coupling
A2 = 0.5  # outflow coefficient, Tank2 -> reservoir drain
A3 = 0.5  # outflow coefficient, Tank3<->Tank2 coupling
TAU_T = 60.0  # baseline thermal time constant, s
K_MIX = 6.0  # inflow-mixing gain on thermal response

# Register-name -> physics-state attribute (the cabinet's domain semantics).
LEVEL_ATTR = {"tank1_level": "h1", "tank2_level": "h2", "tank3_level": "h3"}
TEMP_ATTR = {"tank1_temp": "T1", "tank2_temp": "T2", "tank3_temp": "T3"}
# Actuator-name -> pump index (order of the q tuple passed to step()).
ACTUATOR_Q = {"actuator1": 0, "actuator2": 1}

# Episode reset (driven by the env between episodes via reset_cmd + init_h*).
RESET_CMD = "reset_cmd"
INIT_LEVEL_ATTR = {"init_h1": "h1", "init_h2": "h2", "init_h3": "h3"}
LEVEL_DEFAULTS = {"h1": 0.30, "h2": 0.18, "h3": 0.24}  # when an init_h* register is 0
RESET_TEMP_C = 24.0  # warm-start temperature after a reset

LOG = logging.getLogger("mock_cabinet")


# --------------------------------------------------------------------------- #
# Contract loading
# --------------------------------------------------------------------------- #
def load_contract(path: str | Path | None = None) -> dict:
    with open(path or CONFIG_PATH) as fh:
        return json.load(fh)


@dataclass
class Layout:
    """Register layout derived from the contract (all ordered by address)."""

    names: list[str]            # every register, address order
    addr: dict[str, int]        # name -> 0-based address
    scale: dict[str, float]     # name -> engineering-per-raw
    sensors: list[str]          # direction = "read", address order
    actuators: list[str]        # direction = "write", address order

    @property
    def base(self) -> int:
        return min(self.addr.values())

    @property
    def n(self) -> int:
        return len(self.names)


def derive_layout(contract: dict) -> Layout:
    regs = sorted(contract["registers"], key=lambda r: r["address"])
    return Layout(
        names=[r["name"] for r in regs],
        addr={r["name"]: int(r["address"]) for r in regs},
        scale={r["name"]: float(r["scale"]) for r in regs},
        sensors=[r["name"] for r in regs if r["direction"] == "read"],
        actuators=[r["name"] for r in regs if r["direction"] == "write"],
    )


@dataclass
class PhysicsParams:
    """Contract-sourced process parameters fed to the integrator."""

    q_max: float       # max pump flow, m^3/s
    h_max: float       # tank height (overflow), m
    t_supply: float    # inlet/supply temperature, degC
    act_max: int       # raw full-scale actuator drive (= round(1/scale))

    @classmethod
    def from_contract(cls, contract: dict) -> "PhysicsParams":
        p = contract["process"]
        act_scale = next(r["scale"] for r in contract["registers"]
                         if r["name"] in ACTUATOR_Q)
        return cls(q_max=float(p["q_max_m3s"]), h_max=float(p["h_max_m"]),
                   t_supply=float(p["t_supply_c"]), act_max=round(1.0 / act_scale))


def _flow(h_from: float, h_to: float, coeff: float) -> float:
    """Signed Torricelli volumetric flow (m^3/s) between two tank levels."""
    dh = h_from - h_to
    if abs(dh) < 1e-9:
        return 0.0
    return coeff * S_PIPE * math.copysign(math.sqrt(2.0 * G * abs(dh)), dh)


@dataclass
class TankProcess:
    """State of the three-tank process (SI units)."""

    h1: float = 0.30
    h2: float = 0.18
    h3: float = 0.24
    T1: float = 24.0  # warm start; relaxes toward t_supply (20 C) as pumps run
    T2: float = 24.0
    T3: float = 24.0

    def step(self, cmd1: int, cmd2: int, dt: float, p: PhysicsParams) -> None:
        """Advance one Euler step of `dt` seconds given actuator commands."""
        q1 = max(0.0, min(float(cmd1), p.act_max)) / p.act_max * p.q_max
        q2 = max(0.0, min(float(cmd2), p.act_max)) / p.act_max * p.q_max

        q_12 = _flow(self.h1, self.h2, A1)  # Tank1 -> Tank2
        q_32 = _flow(self.h3, self.h2, A3)  # Tank3 -> Tank2
        q_drain = _flow(self.h2, 0.0, A2)  # Tank2 -> reservoir

        self.h1 += (q1 - q_12) * dt / A_TANK
        self.h3 += (q2 - q_32) * dt / A_TANK
        self.h2 += (q_12 + q_32 - q_drain) * dt / A_TANK
        for attr in ("h1", "h2", "h3"):
            setattr(self, attr, max(0.0, min(getattr(self, attr), p.h_max)))

        # First-order thermal: relax toward t_supply, faster with pump inflow.
        self.T1 += self._dtemp(self.T1, self.h1, q1, dt, p.t_supply)
        self.T2 += self._dtemp(self.T2, self.h2, max(q_12 + q_32, 0.0), dt, p.t_supply)
        self.T3 += self._dtemp(self.T3, self.h3, q2, dt, p.t_supply)
        for attr in ("T1", "T2", "T3"):
            setattr(self, attr, max(0.0, min(getattr(self, attr), 100.0)))

    @staticmethod
    def _dtemp(temp: float, level: float, q_in: float, dt: float, t_supply: float) -> float:
        vol = A_TANK * max(level, 0.02)
        turnover = max(q_in, 0.0) / vol
        return (t_supply - temp) * (1.0 / TAU_T + K_MIX * turnover) * dt

    def snapshot(self) -> dict:
        return {
            "h1_cm": round(self.h1 * 100, 2), "h2_cm": round(self.h2 * 100, 2),
            "h3_cm": round(self.h3 * 100, 2),
            "T1": round(self.T1, 2), "T2": round(self.T2, 2), "T3": round(self.T3, 2),
        }


# --------------------------------------------------------------------------- #
# Encoding (engineering -> raw uint16), driven by the contract scales
# --------------------------------------------------------------------------- #
def encode_sensor(name: str, proc: TankProcess, scale: float, h_max: float) -> int:
    """raw = engineering / scale, clamped to the register's physical range."""
    if name in LEVEL_ATTR:
        eng = max(0.0, min(getattr(proc, LEVEL_ATTR[name]), h_max))
    elif name in TEMP_ATTR:
        eng = max(0.0, min(getattr(proc, TEMP_ATTR[name]), 100.0))
    else:
        raise ValueError(f"don't know how to encode sensor register '{name}'")
    return int(round(eng / scale))


def sensor_registers(proc: TankProcess, layout: Layout, params: PhysicsParams) -> list[int]:
    """Sensor values in address order (the order iomap/async_setValues expect)."""
    return [encode_sensor(s, proc, layout.scale[s], params.h_max) for s in layout.sensors]


def apply_reset(proc: TankProcess, regval: dict, layout: Layout, params: PhysicsParams) -> None:
    """Snap the process to the requested initial state (episode reset).

    Levels come from the init_h* registers (raw -> m via the contract scale); a
    zero/missing init register falls back to the default. Temps return to a warm
    start. Triggered by the rising edge of reset_cmd in physics_loop.
    """
    for reg, attr in INIT_LEVEL_ATTR.items():
        raw = regval.get(reg, 0)
        val = (raw * layout.scale[reg]) if raw and reg in layout.scale else LEVEL_DEFAULTS[attr]
        setattr(proc, attr, max(0.0, min(val, params.h_max)))
    for attr in ("T1", "T2", "T3"):
        setattr(proc, attr, RESET_TEMP_C)


def build_server(host: str, port: int, unit_id: int, proc: TankProcess,
                 layout: Layout, params: PhysicsParams) -> ModbusTcpServer:
    """Create the pymodbus TCP server; initial regs = encoded state + actuators off."""
    initial: list[int] = []
    for name in layout.names:
        if name in LEVEL_ATTR or name in TEMP_ATTR:
            initial.append(encode_sensor(name, proc, layout.scale[name], params.h_max))
        else:  # actuators (and any future non-sensor output) start at 0
            initial.append(0)
    hr = SimData(layout.base, values=initial, datatype=DataType.REGISTERS)
    device = SimDevice(unit_id, simdata=[hr])
    return ModbusTcpServer(context=[device], address=(host, port))


async def physics_loop(
    server: ModbusTcpServer, unit_id: int, proc: TankProcess,
    layout: Layout, params: PhysicsParams, dt: float, log_every: int,
) -> None:
    """Tick: read actuator regs, integrate physics, write sensor regs.

    Assumes the register block is contiguous from `layout.base` (true for the
    current contract; revisit if future registers fragment the address space).
    """
    sensor_base = min(layout.addr[s] for s in layout.sensors)
    tick = 0
    prev_reset_val = 0
    while True:
        all_vals = await server.async_getValues(unit_id, 3, layout.base, layout.n)
        regval = {name: int(v) for name, v in zip(layout.names, all_vals)}

        # Episode reset: a CHANGE to a new nonzero reset_cmd value snaps state
        # to init_h*; while reset_cmd stays nonzero the plant HOLDS that state
        # (no step) so the env can read clean init levels; writing 0 resumes
        # stepping. Value-change (not boolean edge) so back-to-back resets with
        # only a brief 0 between them still trigger — the env writes a fresh
        # nonce each reset.
        reset_val = int(regval.get(RESET_CMD, 0))
        if reset_val != 0 and reset_val != prev_reset_val:
            apply_reset(proc, regval, layout, params)
            LOG.info("reset applied -> %s", proc.snapshot())
        if reset_val == 0:
            proc.step(regval.get("actuator1", 0), regval.get("actuator2", 0), dt, params)
        prev_reset_val = reset_val

        await server.async_setValues(
            unit_id, 16, sensor_base, sensor_registers(proc, layout, params)
        )

        tick += 1
        if log_every and tick % log_every == 0:
            LOG.info("act=[%5d %5d]  %s", regval.get("actuator1", 0),
                     regval.get("actuator2", 0), proc.snapshot())
        await asyncio.sleep(dt)


def _install_signals(loop: asyncio.AbstractEventLoop, server: ModbusTcpServer) -> None:
    async def _stop() -> None:
        LOG.info("shutdown signal received")
        await server.shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(_stop()))


async def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Three-tank Modbus TCP cabinet.")
    parser.add_argument("--config", default=str(CONFIG_PATH),
                        help="path to ia2_config.json (the register contract)")
    parser.add_argument("--host", default=None, help="bind host (default: contract)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default: contract)")
    parser.add_argument("--dt", type=float, default=0.05, help="physics step, seconds")
    parser.add_argument("--log-every", type=int, default=50, help="log every N ticks (0=off)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        logging.getLogger("pymodbus").setLevel(logging.WARNING)

    contract = load_contract(args.config)
    layout = derive_layout(contract)
    params = PhysicsParams.from_contract(contract)
    unit_id = int(contract["modbus"]["unit_id"])
    host = args.host or contract["modbus"]["host"]
    port = args.port or int(contract["modbus"]["port"])

    proc = TankProcess()
    server = build_server(host, port, unit_id, proc, layout, params)
    _install_signals(asyncio.get_running_loop(), server)

    phys = asyncio.create_task(
        physics_loop(server, unit_id, proc, layout, params, args.dt, args.log_every),
        name="physics",
    )
    LOG.info(
        "listening on %s:%d (device_id=%d, %d regs) — %s",
        host, port, unit_id, layout.n, proc.snapshot(),
    )

    try:
        await server.serve_forever()
    finally:
        if not phys.done():
            phys.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await phys
        LOG.info("stopped")


if __name__ == "__main__":
    asyncio.run(main())
