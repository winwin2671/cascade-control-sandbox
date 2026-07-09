#!/usr/bin/env python3
"""mock_cabinet.py — three-tank "decoupled" process simulator (Modbus TCP server).

A pymodbus TCP *server* (slave) bound to 127.0.0.1:5020 that exposes an
8-register holding-register map describing a three-tank hydraulic + thermal
process.  An asyncio physics loop updates the sensor registers (levels and
temperatures) every tick from whatever actuator-command registers a Modbus
master (the IA2 engine, a hand-written controller, or an RL agent) writes.

Modbus register map (holding registers; PLC "4xxxx" notation -> 0-based addr):

    40001 / addr 0 : Tank 1 Level
    40002 / addr 1 : Tank 1 Temperature
    40003 / addr 2 : Tank 2 Level
    40004 / addr 3 : Tank 2 Temperature
    40005 / addr 4 : Tank 3 (Decoupled) Level
    40006 / addr 5 : Tank 3 (Decoupled) Temperature
    40007 / addr 6 : Actuator Command 1   (pump 1 drive)
    40008 / addr 7 : Actuator Command 2   (pump 2 drive)

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

Register encoding (uint16).  These scales MUST match ia2_config.json so IA2
decodes the registers back into engineering units:

    level : reg = level_m  * LEVEL_SCALE      (level_m = reg / LEVEL_SCALE)
    temp  : reg = temp_degC * TEMP_SCALE       (temp_c  = reg / TEMP_SCALE)
    pump  : reg = drive 0..ACT_MAX  (per-mille-of-full-scale x10)
             flow_q = (reg / ACT_MAX) * Q_MAX
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import math
import signal
from dataclasses import dataclass

from pymodbus.server import ModbusTcpServer
from pymodbus.simulator import SimData, SimDevice
from pymodbus.simulator.simutils import DataType

# --------------------------------------------------------------------------- #
# Network / Modbus identity
# --------------------------------------------------------------------------- #
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5020
UNIT_ID = 1  # Modbus device/unit id the master must address

# --------------------------------------------------------------------------- #
# Register addresses (holding registers, 0-based -> PLC 4xxxx)
# --------------------------------------------------------------------------- #
ADDR_T1_LEVEL = 0  # 40001
ADDR_T1_TEMP = 1  # 40002
ADDR_T2_LEVEL = 2  # 40003
ADDR_T2_TEMP = 3  # 40004
ADDR_T3_LEVEL = 4  # 40005  (Tank 3 "Decoupled")
ADDR_T3_TEMP = 5  # 40006
ADDR_ACT1 = 6  # 40007
ADDR_ACT2 = 7  # 40008
N_REGS = 8

# --------------------------------------------------------------------------- #
# Engineering <-> register scales
# --------------------------------------------------------------------------- #
LEVEL_SCALE = 10_000  # 1 LSB = 0.1 mm of level
TEMP_SCALE = 100  # 1 LSB = 0.01 degC
ACT_MAX = 10_000  # full-scale actuator drive (100.00 %)

# --------------------------------------------------------------------------- #
# Process parameters (SI units).  Tuned for a responsive but stable mock.
# --------------------------------------------------------------------------- #
G = 9.81  # gravity, m/s^2
A_TANK = 0.0154  # tank cross-sectional area, m^2
S_PIPE = 5.0e-5  # connecting-pipe cross-section, m^2
A1 = 0.5  # outflow coefficient, Tank1<->Tank2 coupling
A2 = 0.5  # outflow coefficient, Tank2 -> reservoir drain
A3 = 0.5  # outflow coefficient, Tank3<->Tank2 coupling
H_MAX = 0.60  # tank height (overflow), m
Q_MAX = 2.0e-4  # max pump flow per pump, m^3/s
T_SUPPLY = 20.0  # inlet/supply water temperature, degC
TAU_T = 60.0  # baseline thermal time constant, s
K_MIX = 6.0  # inflow-mixing gain on thermal response

LOG = logging.getLogger("mock_cabinet")


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
    T1: float = 24.0  # warm start; relaxes toward T_SUPPLY (20 C) as pumps run
    T2: float = 24.0
    T3: float = 24.0

    def step(self, cmd1: int, cmd2: int, dt: float) -> None:
        """Advance one Euler step of `dt` seconds given actuator commands."""
        q1 = max(0.0, min(float(cmd1), ACT_MAX)) / ACT_MAX * Q_MAX
        q2 = max(0.0, min(float(cmd2), ACT_MAX)) / ACT_MAX * Q_MAX

        q_12 = _flow(self.h1, self.h2, A1)  # Tank1 -> Tank2
        q_32 = _flow(self.h3, self.h2, A3)  # Tank3 -> Tank2
        q_drain = _flow(self.h2, 0.0, A2)  # Tank2 -> reservoir

        self.h1 += (q1 - q_12) * dt / A_TANK
        self.h3 += (q2 - q_32) * dt / A_TANK
        self.h2 += (q_12 + q_32 - q_drain) * dt / A_TANK
        for attr in ("h1", "h2", "h3"):
            setattr(self, attr, max(0.0, min(getattr(self, attr), H_MAX)))

        # First-order thermal: relax toward T_SUPPLY, faster with pump inflow.
        self.T1 += self._dtemp(self.T1, self.h1, q1, dt)
        self.T2 += self._dtemp(self.T2, self.h2, max(q_12 + q_32, 0.0), dt)
        self.T3 += self._dtemp(self.T3, self.h3, q2, dt)
        for attr in ("T1", "T2", "T3"):
            setattr(self, attr, max(0.0, min(getattr(self, attr), 100.0)))

    @staticmethod
    def _dtemp(temp: float, level: float, q_in: float, dt: float) -> float:
        vol = A_TANK * max(level, 0.02)
        turnover = max(q_in, 0.0) / vol
        return (T_SUPPLY - temp) * (1.0 / TAU_T + K_MIX * turnover) * dt

    def snapshot(self) -> dict:
        return {
            "h1_cm": round(self.h1 * 100, 2),
            "h2_cm": round(self.h2 * 100, 2),
            "h3_cm": round(self.h3 * 100, 2),
            "T1": round(self.T1, 2),
            "T2": round(self.T2, 2),
            "T3": round(self.T3, 2),
        }


def _enc_level(m: float) -> int:
    return int(round(max(0.0, min(m, H_MAX)) * LEVEL_SCALE))


def _enc_temp(c: float) -> int:
    return int(round(max(0.0, min(c, 100.0)) * TEMP_SCALE))


def sensor_registers(proc: TankProcess) -> list[int]:
    """Encode the six sensor values into holding-register uint16s (addr 0..5)."""
    return [
        _enc_level(proc.h1), _enc_temp(proc.T1),
        _enc_level(proc.h2), _enc_temp(proc.T2),
        _enc_level(proc.h3), _enc_temp(proc.T3),
    ]


def build_server(host: str, port: int, proc: TankProcess) -> ModbusTcpServer:
    """Create the pymodbus TCP server with a single device holding 8 registers."""
    initial = sensor_registers(proc) + [0, 0]  # actuators start OFF
    hr = SimData(0, values=initial, datatype=DataType.REGISTERS)
    device = SimDevice(UNIT_ID, simdata=[hr])
    server = ModbusTcpServer(context=[device], address=(host, port))
    # server.context is the live SimCore; physics updates go through
    # server.async_getValues / server.async_setValues (the public API).
    return server


async def physics_loop(
    server: ModbusTcpServer, proc: TankProcess, dt: float, log_every: int
) -> None:
    """Tick the process: read actuator regs, integrate physics, write sensor regs."""
    tick = 0
    while True:
        vals = await server.async_getValues(UNIT_ID, 3, ADDR_ACT1, 2)
        cmd1, cmd2 = int(vals[0]), int(vals[1])

        proc.step(cmd1, cmd2, dt)
        await server.async_setValues(UNIT_ID, 16, ADDR_T1_LEVEL, sensor_registers(proc))

        tick += 1
        if log_every and tick % log_every == 0:
            LOG.info(
                "act=[%5d %5d]  %s",
                cmd1, cmd2, proc.snapshot(),
            )
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
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
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

    proc = TankProcess()
    server = build_server(args.host, args.port, proc)
    _install_signals(asyncio.get_running_loop(), server)

    phys = asyncio.create_task(
        physics_loop(server, proc, args.dt, args.log_every), name="physics"
    )
    LOG.info(
        "listening on %s:%d (device_id=%d) — %s",
        args.host, args.port, UNIT_ID, proc.snapshot(),
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
