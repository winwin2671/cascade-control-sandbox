#!/usr/bin/env python3
"""mock_cabinet.py — heated serial-cascade + recirculation process simulator (Modbus).

A pymodbus TCP server bound to 127.0.0.1:5020 that emulates the field cabinet
behind the RTU-to-TCP gateway described in ia2_config.json. It serves MULTIPLE
Modbus slaves (one SimDevice per unit_id) with segregated function codes:

    slave 2  AI     FC04 input registers  (3 levels + 2 temps + 3 flows, float)
    slave 5  AI #2  FC04 input registers  (TT-301, float)
    slave 3  AO+    FC06/FC16 holding     (V-12, V-23, E-101 cmds [float] + reset [uint16])
    slave 4  DI     FC02 discrete inputs  (5 hardware-safety-status flags)
    slave 6  VFD    FC16 holding          (P-101 frequency cmd, float)

Process topology — heated serial cascade with recirculation:

    pump P-101 --> Tank 1 --(prop. valve V-12)--> Tank 2 --(prop. valve V-23)--> Tank 3
                                              Tank 3 --(manual valve V-3)--> Reservoir
                                              Reservoir --(P-101)--> Tank 1   (recirculation)
    heater E-101 (2 kW) --> Tank 1 only

Tank1 is the only directly-heated tank; Tank2/Tank3 warm via downstream hot-water
advection (the under-actuated temperature coupling). Reservoir is modeled as
infinite (constant level, constant temp = supply).

Config-driven. The register layout (names, slave_ids, addresses, function codes,
data types, byte/word order) is read from ia2_config.json — the same single
contract the IA2 iomap and aio_bridge_env.py use. 32-bit floats are stored as
two big-endian (ABCD) registers via pack_float_be.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import signal
import struct
from dataclasses import dataclass
from pathlib import Path

from pymodbus.server import ModbusTcpServer
from pymodbus.simulator import SimData, SimDevice
from pymodbus.simulator.simutils import DataType

CONFIG_PATH = Path(__file__).resolve().parent / "ia2_config.json"

# --------------------------------------------------------------------------- #
# Physics constants (gravity only is universal; geometry/coeffs come from the
# contract process block — they are PLACEHOLDERS until the rig is measured).
# --------------------------------------------------------------------------- #
G = 9.81  # gravity, m/s^2

# Register-name -> process-state attribute.
LEVEL_ATTR = {"tank1_level": "h1", "tank2_level": "h2", "tank3_level": "h3"}
TEMP_ATTR = {"tank1_temp": "T1", "tank2_temp": "T2", "tank3_temp": "T3"}
FLOW_ATTR = {"tank1_flow": "q12_lpm", "tank2_flow": "q23_lpm", "tank3_flow": "q3r_lpm"}
DI_ATTR = {"di_dryfire": "di_dryfire", "di_overflow": "di_overflow",
           "di_heater_contactor": "di_heater_contactor",
           "di_pump_contactor": "di_pump_contactor", "di_estop": "di_estop"}

# Actuator command register names (decoded from their slave holding blocks).
VFD_CMD, V12_CMD, V23_CMD, V33_CMD, E101_CMD = "vfd_cmd", "v_12_cmd", "v_23_cmd", "v_33_cmd", "e_101_cmd"


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


# Episode reset (driven by the env between episodes via reset_cmd + init_h*).
RESET_CMD = "reset_cmd"
INIT_LEVEL_ATTR = {"init_h1": "h1", "init_h2": "h2", "init_h3": "h3"}
LEVEL_DEFAULTS = {"h1": 0.30, "h2": 0.22, "h3": 0.18}  # when an init_h* register is 0
RESET_TEMP_C = 25.0  # warm-start temperature after a reset (= supply/ambient)

LOG = logging.getLogger("mock_cabinet")


# --------------------------------------------------------------------------- #
# 32-bit float register codec (Modbus "ABCD": big-endian bytes, high word first).
# MUST match aio_bridge_env.decode_float — the mock encodes, the bridge decodes.
# --------------------------------------------------------------------------- #
def pack_float_be(v: float) -> list[int]:
    """Encode a float to two big-endian (ABCD) 16-bit registers."""
    b = struct.pack(">f", float(v))
    return [int.from_bytes(b[0:2], "big"), int.from_bytes(b[2:4], "big")]


def unpack_float_be(regs) -> float:
    """Decode two big-endian (ABCD) 16-bit registers back to a float."""
    r0, r1 = int(regs[0]), int(regs[1])
    raw = bytes(((r0 >> 8) & 0xFF, r0 & 0xFF, (r1 >> 8) & 0xFF, r1 & 0xFF))
    return struct.unpack(">f", raw)[0]


# --------------------------------------------------------------------------- #
# Contract loading + layout
# --------------------------------------------------------------------------- #
def load_contract(path: str | Path | None = None) -> dict:
    with open(path or CONFIG_PATH) as fh:
        return json.load(fh)


@dataclass
class Layout:
    """Register layout derived from the contract, grouped for multi-slave I/O."""
    regs: list[dict]                       # all registers, sorted by (slave_id, address)
    by_name: dict[str, dict]
    sensors: list[dict]                    # direction = "read"
    actuators: list[dict]                  # direction = "write", group = "actuators"
    by_slave: dict[int, list[dict]]        # slave_id -> its registers (address order)
    holding_by_slave: dict[int, list[dict]]   # slaves the physics loop must READ (cmds/reset)
    publish_by_slave: dict[int, list[dict]]   # slaves the physics loop must WRITE (input/discrete)


def derive_layout(contract: dict) -> "Layout":
    regs = sorted(contract["registers"], key=lambda r: (r["slave_id"], r["address"]))
    by_slave: dict[int, list[dict]] = {}
    for r in regs:
        by_slave.setdefault(r["slave_id"], []).append(r)
    holding_by_slave = {sid: [r for r in rs if r["table"] == "holding"] for sid, rs in by_slave.items()}
    holding_by_slave = {sid: rs for sid, rs in holding_by_slave.items() if rs}
    publish_by_slave = {
        sid: [r for r in rs if r["table"] in ("input", "discrete_input")]
        for sid, rs in by_slave.items()
    }
    publish_by_slave = {sid: rs for sid, rs in publish_by_slave.items() if rs}
    return Layout(
        regs=regs,
        by_name={r["name"]: r for r in regs},
        sensors=[r for r in regs if r["direction"] == "read"],
        actuators=[r for r in regs if r["direction"] == "write" and r.get("group") == "actuators"],
        by_slave=by_slave,
        holding_by_slave=holding_by_slave,
        publish_by_slave=publish_by_slave,
    )


@dataclass
class PhysicsParams:
    """Contract-sourced process parameters fed to the integrator."""
    q_max: float        # max pump flow at full VFD, m^3/s
    vfd_max_hz: float   # VFD frequency at full pump speed, Hz
    h_max: float        # tank height (overflow), m
    t_supply: float     # reservoir/supply temperature, degC
    a_tank: float       # tank cross-section, m^2
    c_v12: float        # V-12 effective orifice (C_d * area), m^2
    c_v23: float        # V-23 effective orifice, m^2
    c_v33: float        # V-33 effective orifice (Tank3 -> reservoir), m^2
    cp: float           # specific heat capacity, J/(kg.K)
    rho: float          # water density, kg/m^3
    q_heat_max: float   # max electrical heater power (E-101, 2 kW), W
    ua: float           # overall heat-loss coefficient, W/K
    t_ambient: float    # ambient (heat-sink) temperature, degC

    @classmethod
    def from_contract(cls, contract: dict) -> "PhysicsParams":
        p = contract["process"]
        return cls(
            q_max=float(p["q_max_m3s"]), vfd_max_hz=float(p["vfd_max_hz"]),
            h_max=float(p["h_max_m"]), t_supply=float(p["t_supply_c"]),
            a_tank=float(p["a_tank_m2"]),
            c_v12=float(p["c_v12"]), c_v23=float(p["c_v23"]), c_v33=float(p["c_v33"]),
            cp=float(p["cp_j_per_kgk"]), rho=float(p["rho_kg_per_m3"]),
            q_heat_max=float(p["q_heat_max_w"]), ua=float(p["ua_w_per_k"]),
            t_ambient=float(p["t_ambient_c"]),
        )


def _valve_flow(h_from: float, h_to: float, c_v: float, frac: float) -> float:
    """Unidirectional, valve-modulated Torricelli volumetric flow (m^3/s).

    Only flows downhill (h_from > h_to); `frac` in [0,1] is the valve position
    (1.0 for the manual V-3 which is fixed open). `c_v` is the effective orifice.
    """
    dh = h_from - h_to
    if dh <= 1e-9:
        return 0.0
    return _clamp(frac, 0.0, 1.0) * c_v * math.sqrt(2.0 * G * dh)


@dataclass
class TankProcess:
    """State of the heated serial-cascade process (SI units)."""

    h1: float = 0.30
    h2: float = 0.22
    h3: float = 0.18
    T1: float = 25.0
    T2: float = 25.0
    T3: float = 25.0
    # last-computed inter-tank flows (L/min) — published to the FT sensors
    q12_lpm: float = 0.0
    q23_lpm: float = 0.0
    q3r_lpm: float = 0.0
    # emulated hardware safety-status flags (the real ones are hardwired relays)
    di_dryfire: int = 0
    di_overflow: int = 0
    di_heater_contactor: int = 0
    di_pump_contactor: int = 0
    di_estop: int = 0

    def step(self, vfd_hz: float, v12_pct: float, v23_pct: float, v33_pct: float,
             e101_pct: float, dt: float, p: PhysicsParams) -> None:
        """Advance one Euler step given the 4 actuator commands.

        Hydraulics: pump recirculation into Tank1 + unidirectional valve-modulated
        Torricelli cascade T1->T2->T3->reservoir. Thermal: first law per tank
        (m.cp.dT/dt = Q_heat - Q_loss + advection) with the heater only in Tank1;
        Tank2/Tank3 warm solely via downstream hot-water advection.
        """
        # --- hydraulics ---
        vfd_frac = _clamp(vfd_hz / p.vfd_max_hz, 0.0, 1.0)
        q_pump = vfd_frac * p.q_max                                   # P-101 -> Tank1
        v12_frac = _clamp(v12_pct / 100.0, 0.0, 1.0)
        v23_frac = _clamp(v23_pct / 100.0, 0.0, 1.0)
        v33_frac = _clamp(v33_pct / 100.0, 0.0, 1.0)
        q_12 = _valve_flow(self.h1, self.h2, p.c_v12, v12_frac)       # Tank1 -> Tank2
        q_23 = _valve_flow(self.h2, self.h3, p.c_v23, v23_frac)       # Tank2 -> Tank3
        q_3r = _valve_flow(self.h3, 0.0, p.c_v33, v33_frac)           # Tank3 -> reservoir (V-33 control)

        self.h1 += (q_pump - q_12) * dt / p.a_tank
        self.h2 += (q_12 - q_23) * dt / p.a_tank
        self.h3 += (q_23 - q_3r) * dt / p.a_tank
        for attr in ("h1", "h2", "h3"):
            setattr(self, attr, _clamp(getattr(self, attr), 0.0, p.h_max))

        self.q12_lpm = q_12 * 60000.0
        self.q23_lpm = q_23 * 60000.0
        self.q3r_lpm = q_3r * 60000.0

        # --- thermal: single heater in Tank1; chain advection ---
        Qh1 = _clamp(e101_pct / 100.0, 0.0, 1.0) * p.q_heat_max
        adv1 = q_pump * (p.t_supply - self.T1)      # recirc returns reservoir-temp water to Tank1
        adv2 = q_12 * (self.T1 - self.T2)           # hot Tank1 outflow -> Tank2
        adv3 = q_23 * (self.T2 - self.T3)           # Tank2 outflow -> Tank3 (Tank3 loses via q_3r: outflow drops out)
        self._step_thermal("T1", "h1", Qh1, adv1, dt, p)
        self._step_thermal("T2", "h2", 0.0, adv2, dt, p)
        self._step_thermal("T3", "h3", 0.0, adv3, dt, p)

        # --- emulated safety flags (real ones are hardwired; mock reflects state) ---
        self.di_dryfire = 1 if self.h1 < 0.05 else 0
        self.di_overflow = 1 if max(self.h1, self.h2, self.h3) > 0.45 else 0
        self.di_heater_contactor = 1 if Qh1 > 0.0 else 0
        self.di_pump_contactor = 1 if q_pump > 0.0 else 0
        self.di_estop = 0

    def _step_thermal(self, t_attr: str, h_attr: str, q_heat: float,
                      adv: float, dt: float, p: PhysicsParams) -> None:
        T = getattr(self, t_attr)
        h = max(getattr(self, h_attr), 0.02)        # guard against empty-tank /0
        m_cp = p.rho * p.a_tank * h * p.cp           # thermal capacitance, J/K
        q_loss = p.ua * (T - p.t_ambient)            # Newton cooling, W
        dT = (q_heat - q_loss) / m_cp * dt + adv / (p.a_tank * h) * dt
        setattr(self, t_attr, _clamp(T + dT, 0.0, 100.0))

    def snapshot(self) -> dict:
        return {
            "h_cm": [round(self.h1 * 100, 2), round(self.h2 * 100, 2), round(self.h3 * 100, 2)],
            "T": [round(self.T1, 2), round(self.T2, 2), round(self.T3, 2)],
            "flow_lpm": [round(self.q12_lpm, 2), round(self.q23_lpm, 2), round(self.q3r_lpm, 2)],
        }


# --------------------------------------------------------------------------- #
# Encoding (engineering -> raw register values), driven by the contract
# --------------------------------------------------------------------------- #
def encode_channel(reg: dict, proc: TankProcess, params: PhysicsParams) -> list:
    """Raw values for one read channel: 2 ints (float), 1 int (uint16), or 1 bool (DI)."""
    name = reg["name"]
    if reg["type"] == "bool":
        return [bool(getattr(proc, DI_ATTR[name]))]   # BITS demands bools, not ints
    if name in LEVEL_ATTR:
        eng = _clamp(getattr(proc, LEVEL_ATTR[name]), 0.0, params.h_max)
    elif name in TEMP_ATTR:
        eng = _clamp(getattr(proc, TEMP_ATTR[name]), 0.0, 100.0)
    elif name in FLOW_ATTR:
        eng = _clamp(getattr(proc, FLOW_ATTR[name]), 0.0, float(reg.get("max", 50.0)))
    else:
        return [0] * int(reg.get("count", 1))
    return pack_float_be(eng)


def apply_reset(proc: TankProcess, regval: dict, layout: Layout, params: PhysicsParams) -> None:
    """Snap the process to the requested initial state (episode reset).

    Levels come from the init_h* registers (raw uint16 -> m via the contract
    scale); a zero/missing init register falls back to the default. Temps return
    to a warm start. Triggered by a value-change of reset_cmd in physics_loop.
    """
    for reg, attr in INIT_LEVEL_ATTR.items():
        raw = regval.get(reg, 0)
        r = layout.by_name.get(reg)
        scale = float(r["scale"]) if r else 0.0001
        val = (raw * scale) if raw else LEVEL_DEFAULTS[attr]
        setattr(proc, attr, _clamp(val, 0.0, params.h_max))
    for attr in ("T1", "T2", "T3"):
        setattr(proc, attr, RESET_TEMP_C)


def _decode_holding(raw_vals: list[int], regs: list[dict]) -> dict:
    """Decode a contiguous holding block into {name: value} (float/uint16)."""
    out: dict = {}
    base = regs[0]["address"]
    for r in regs:
        off = r["address"] - base
        chunk = raw_vals[off:off + int(r["count"])]
        if r["type"] == "float":
            out[r["name"]] = unpack_float_be(chunk)
        elif r["type"] == "uint16":
            out[r["name"]] = int(chunk[0])
        else:
            out[r["name"]] = chunk[0]
    return out


def _simdata_blocks(regs_on_slave: list[dict], proc: TankProcess, params: PhysicsParams):
    """Build the (coils, discrete_in, holding, input_reg) SimData 4-tuple for a slave.

    pymodbus requires ALL FOUR tuple slots to be non-empty (its block checker
    indexes [0]), so empty tables get a 1-value dummy block. Real read tables
    (input/discrete) are seeded with the encoded process state; write tables
    (holding) start at zero. Note SimData `count` *repeats* the values list, so
    for a list of N distinct register values we leave count at its default (1).
    """
    by_table: dict[str, list[dict]] = {}
    for r in regs_on_slave:
        by_table.setdefault(r["table"], []).append(r)

    def block(table: str, datatype, is_read: bool) -> list:
        rs = sorted(by_table.get(table, []), key=lambda r: r["address"])
        if not rs:
            dummy = False if datatype == DataType.BITS else 0
            return [SimData(0, values=[dummy], datatype=datatype)]  # non-empty placeholder
        vals: list = []
        for r in rs:
            vals += encode_channel(r, proc, params) if is_read else [0] * int(r["count"])
        return [SimData(rs[0]["address"], values=vals, datatype=datatype)]  # count defaults to 1

    return (block("coil", DataType.BITS, True),
            block("discrete_input", DataType.BITS, True),
            block("holding", DataType.REGISTERS, False),
            block("input", DataType.REGISTERS, True))


def build_server(host: str, port: int, proc: TankProcess,
                 layout: Layout, params: PhysicsParams) -> ModbusTcpServer:
    """One SimDevice per slave (multi-slave on a single TCP listener)."""
    devices = []
    for slave_id in sorted(layout.by_slave):
        c, d, h, i = _simdata_blocks(layout.by_slave[slave_id], proc, params)
        devices.append(SimDevice(slave_id, simdata=(c, d, h, i)))
    return ModbusTcpServer(context=devices, address=(host, port))


async def physics_loop(
    server: ModbusTcpServer, proc: TankProcess, layout: Layout,
    params: PhysicsParams, dt: float, log_every: int, time_scale: float = 1.0,
) -> None:
    """Tick: read actuator/reset regs (per slave), integrate physics, publish sensors.

    `dt` is plant-seconds per tick; `time_scale` > 1 compresses wall-clock
    (training). Reads union every slave's holding block so reset_cmd (slave 03)
    is visible alongside the VFD (slave 06) and valve/heater cmds (slave 03).
    """
    tick = 0
    prev_reset_val = 0
    wall_budget = dt / time_scale
    next_deadline = asyncio.get_event_loop().time() + wall_budget

    # Pre-compute the holding read plan (slave_id -> (base, count, sorted regs)).
    holding_plan = {}
    for sid, regs in layout.holding_by_slave.items():
        regs = sorted(regs, key=lambda r: r["address"])
        holding_plan[sid] = (regs[0]["address"], sum(int(r["count"]) for r in regs), regs)

    while True:
        # --- read all holding (actuator + reset) blocks, union into regval ---
        regval: dict = {}
        for sid, (base, count, regs) in holding_plan.items():
            raw = await server.async_getValues(sid, 3, base, count)
            regval.update(_decode_holding(raw, regs))

        # --- episode reset (nonce value-change on reset_cmd, slave 03) ---
        reset_val = int(regval.get(RESET_CMD, 0))
        if reset_val != 0 and reset_val != prev_reset_val:
            apply_reset(proc, regval, layout, params)
            LOG.info("reset applied -> %s", proc.snapshot())
        if reset_val == 0:
            proc.step(regval.get(VFD_CMD, 0.0), regval.get(V12_CMD, 0.0),
                      regval.get(V23_CMD, 0.0), regval.get(V33_CMD, 0.0),
                      regval.get(E101_CMD, 0.0), dt, params)
        prev_reset_val = reset_val

        # --- publish sensors + DI flags to each slave's read blocks ---
        for sid, regs in layout.publish_by_slave.items():
            by_table: dict[str, list[dict]] = {}
            for r in regs:
                by_table.setdefault(r["table"], []).append(r)
            for table, fc in (("input", 4), ("discrete_input", 2)):
                if table not in by_table:
                    continue
                rs = sorted(by_table[table], key=lambda r: r["address"])
                vals: list[int] = []
                for r in rs:
                    vals += encode_channel(r, proc, params)
                await server.async_setValues(sid, fc, rs[0]["address"], vals)

        tick += 1
        if log_every and tick % log_every == 0:
            LOG.info("vfd=%5.1fHz v12=%5.1f v23=%5.1f v33=%5.1f e101=%5.1f  %s",
                     regval.get(VFD_CMD, 0.0), regval.get(V12_CMD, 0.0),
                     regval.get(V23_CMD, 0.0), regval.get(V33_CMD, 0.0),
                     regval.get(E101_CMD, 0.0), proc.snapshot())

        now = asyncio.get_event_loop().time()
        remaining = next_deadline - now
        if remaining > 0:
            await asyncio.sleep(remaining)
        next_deadline += wall_budget


def _install_signals(loop: asyncio.AbstractEventLoop, server: ModbusTcpServer) -> None:
    async def _stop() -> None:
        LOG.info("shutdown signal received")
        await server.shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(_stop()))


async def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Heated serial-cascade Modbus cabinet.")
    parser.add_argument("--config", default=str(CONFIG_PATH),
                        help="path to ia2_config.json (the register contract)")
    parser.add_argument("--host", default=None, help="bind host (default: contract)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default: contract)")
    parser.add_argument("--dt", type=float, default=0.05, help="physics step, seconds (plant time)")
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="run this many x faster than real-time (training; default 1)")
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
    host = args.host or contract["modbus"]["host"]
    port = args.port or int(contract["modbus"]["port"])

    proc = TankProcess()
    server = build_server(host, port, proc, layout, params)
    _install_signals(asyncio.get_running_loop(), server)

    phys = asyncio.create_task(
        physics_loop(server, proc, layout, params, args.dt, args.log_every, args.time_scale),
        name="physics",
    )
    LOG.info(
        "listening on %s:%d — %d slaves (%s), %dx — %s",
        host, port, len(layout.by_slave),
        ",".join(str(s) for s in sorted(layout.by_slave)), args.time_scale, proc.snapshot(),
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
