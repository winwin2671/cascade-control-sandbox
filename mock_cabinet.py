#!/usr/bin/env python3
"""mock_cabinet.py — heated serial-cascade + recirculation process simulator (Modbus).

A pymodbus TCP server bound to 127.0.0.1:5020 that emulates the field cabinet
behind the RTU-to-TCP gateway described in ia2_config.json. It serves MULTIPLE
Modbus slaves (one SimDevice per unit_id) with segregated function codes:

    slave 2  AI     FC04 input registers  (3 levels + 2 temps + 3 flows, float)
    slave 5  AI #2  FC04 input registers  (TT-301, float)
    slave 3  AO     FC06 holding, u16     (V-12, V-23, E-101, V-33 cmds, raw 0..10000)
    slave 1  SIM    FC06 holding, u16     (reset_cmd + init_h1-3 — sim-only, not on
                                            the real gateway; kept off the AO card)
    slave 4  DI     FC02 discrete inputs  (5 hardware-safety-status flags)
    slave 6  VFD    FC06 holding, u16     (P-101 duty, 0..10000 = 0..100% of F0-10)
    slave 7  DO     FC05 coils, bool      (SV-1..3 on/off interlock-test solenoids,
                                            each parallel to V-12/V-23/V-33)

Process topology — heated serial cascade with recirculation:

    pump P-101 --> Tank 1 --(prop. valve V-12)--> Tank 2 --(prop. valve V-23)--> Tank 3
                                              Tank 3 --(manual valve V-3)--> Reservoir
                                              Reservoir --(P-101)--> Tank 1   (recirculation)
    heater E-101 (2 kW) --> Tank 1 only

Tank1 is the only directly-heated tank; Tank2/Tank3 warm via downstream hot-water
advection (the under-actuated temperature coupling). Reservoir is finite (level
and temp are integrated states; tank overflows spill back into it).

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
# contract process block — tank geometry + pump curve are from the BOM/
# datasheets; only c_v* and ua_w_per_k remain physics estimates pending SAT).
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

# On/off interlock-test solenoid coils (decoded from the DO module's coil block).
SV1_CMD, SV2_CMD, SV3_CMD = "sv_1_cmd", "sv_2_cmd", "sv_3_cmd"


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
    coil_by_slave: dict[int, list[dict]]      # slaves with FC05 coils (test solenoids)


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
    coil_by_slave = {
        sid: sorted([r for r in rs if r["table"] == "coil"], key=lambda r: r["address"])
        for sid, rs in by_slave.items()
    }
    coil_by_slave = {sid: rs for sid, rs in coil_by_slave.items() if rs}
    return Layout(
        regs=regs,
        by_name={r["name"]: r for r in regs},
        sensors=[r for r in regs if r["direction"] == "read"],
        actuators=[r for r in regs if r["direction"] == "write" and r.get("group") == "actuators"],
        by_slave=by_slave,
        holding_by_slave=holding_by_slave,
        publish_by_slave=publish_by_slave,
        coil_by_slave=coil_by_slave,
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
    cv_sv: float        # SV-1..3 on/off solenoid orifices (test bypasses), m^2
    cp: float           # specific heat capacity, J/(kg.K)
    rho: float          # water density, kg/m^3
    q_heat_max: float   # max electrical heater power (E-101, 2 kW), W
    ua: float           # overall heat-loss coefficient, W/K
    t_ambient: float    # ambient (heat-sink) temperature, degC
    reservoir_base: float    # reservoir cross-section (0.6 x 0.5 m), m^2
    reservoir_height: float  # reservoir wall height, m
    gravity_drop: float      # fixed elevation head between tanks (0 = same elevation), m
    high_level_trip: float   # hardware LSH float trip (DI overflow threshold), m
    low_level_trip: float    # hardware LSL float trip (DI dry-fire threshold), m
    temperature_trip: float  # L4 capillary thermostat trip, degC
    pump_power_max: float    # pump electrical power (energy KPI), W
    nominal_level: float     # nominal operating level, m
    pump_static_head: float  # pump curve: head at rated flow, m
    pump_shutoff_head: float # pump curve: head at zero flow (shutoff), m
    overflow_level: float    # level that opens the hydraulic overflow path, m
    cv_overflow: float       # overflow path flow coefficient
    overflow_head_floor: float  # numerical floor for overflow head

    @classmethod
    def from_contract(cls, contract: dict) -> "PhysicsParams":
        p = contract["process"]
        return cls(
            q_max=float(p["q_max_m3s"]), vfd_max_hz=float(p["vfd_max_hz"]),
            h_max=float(p["h_max_m"]), t_supply=float(p["t_supply_c"]),
            a_tank=float(p["a_tank_m2"]),
            c_v12=float(p["c_v12"]), c_v23=float(p["c_v23"]), c_v33=float(p["c_v33"]),
            cv_sv=float(p.get("c_sv", 0.00143)),
            cp=float(p["cp_j_per_kgk"]), rho=float(p["rho_kg_per_m3"]),
            q_heat_max=float(p["q_heat_max_w"]), ua=float(p["ua_w_per_k"]),
            t_ambient=float(p["t_ambient_c"]),
            reservoir_base=float(p.get("reservoir_base_m2", 0.30)),
            reservoir_height=float(p.get("reservoir_height_m", 0.50)),
            gravity_drop=float(p.get("gravity_drop_m", 0.0)),
            high_level_trip=float(p.get("high_level_trip_m", 0.45)),
            low_level_trip=float(p.get("low_level_trip_m", 0.05)),
            temperature_trip=float(p.get("temperature_trip_c", 70.0)),
            pump_power_max=float(p.get("pump_power_max_w", 370.0)),
            nominal_level=float(p.get("nominal_level_m", 0.30)),
            pump_static_head=float(p.get("pump_static_head_m", 1.7)),
            pump_shutoff_head=float(p.get("pump_shutoff_head_m", 10.0)),
            overflow_level=float(p.get("overflow_level_m", 0.46)),
            cv_overflow=float(p.get("cv_overflow", 0.001)),
            overflow_head_floor=float(p.get("overflow_head_floor", 1e-9)),
        )



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
    di_estop: int = 1     # NC chain: 1 = healthy, 0 = pressed (B1/#6 — the mock
                          # must publish the same polarity the hardware will)
    # reservoir (finite, internal — NOT instrumented; affects pump inflow temp)
    h_res: float = 0.30
    T_res: float = 25.0

    def step(self, vfd_cmd: int, v12_cmd: int, v23_cmd: int, v33_cmd: int,
             e101_cmd: int, dt: float, p: PhysicsParams,
             sv: tuple = (False, False, False)) -> None:
        """Advance one Euler step given the 4 actuator commands + SV coil states.

        Physics delegates to the shared threetank_dynamics module so the mock,
        MPC model, and NMPC oracle use ONE set of equations — no drift. What stays
        mock-only: u16->fraction conversion, Euler integration + state clamps, the
        FT sensor publication, and the emulated hardware DI flags.
        """
        from controllers.threetank_dynamics import (
            dynamics, compute_flows, build_params, NUMERIC_OPS)

        pump_frac = _clamp(vfd_cmd / 10000.0, 0.0, 1.0)
        valve_fracs = [_clamp(v12_cmd / 10000.0, 0.0, 1.0),
                       _clamp(v23_cmd / 10000.0, 0.0, 1.0),
                       _clamp(v33_cmd / 10000.0, 0.0, 1.0)]
        heater_frac = _clamp(e101_cmd / 10000.0, 0.0, 1.0)
        sv_fracs = [1.0 if s else 0.0 for s in sv]

        # flat dynamics params are stable for the cabinet's lifetime; cache them.
        dp = getattr(self, "_dyn_params", None)
        if dp is None or getattr(self, "_dyn_params_id", None) != id(p):
            dp = build_params(p)
            dp["t_ambient"] = p.t_ambient
            self._dyn_params = dp
            self._dyn_params_id = id(p)

        x = [self.h1, self.T1, self.h2, self.T2, self.h3, self.T3, self.h_res, self.T_res]
        flows = compute_flows(x, pump_frac, valve_fracs, heater_frac, dp, NUMERIC_OPS, sv_fracs)
        dx = dynamics(x, pump_frac, valve_fracs, heater_frac, dp, NUMERIC_OPS, sv_fracs)

        # Euler integration + clamps (levels bounded by tank height; temps by 0..100).
        self.h1 = _clamp(self.h1 + dx[0] * dt, 0.0, p.h_max)
        self.T1 = _clamp(self.T1 + dx[1] * dt, 0.0, 100.0)
        self.h2 = _clamp(self.h2 + dx[2] * dt, 0.0, p.h_max)
        self.T2 = _clamp(self.T2 + dx[3] * dt, 0.0, 100.0)
        self.h3 = _clamp(self.h3 + dx[4] * dt, 0.0, p.h_max)
        self.T3 = _clamp(self.T3 + dx[5] * dt, 0.0, 100.0)
        self.h_res = _clamp(self.h_res + dx[6] * dt, 0.0, p.reservoir_height)
        self.T_res = _clamp(self.T_res + dx[7] * dt, 0.0, 100.0)

        # FT sensors (m^3/s -> L/min) from the shared flows. Each FT reads the
        # COMBINED inter-tank line (valve + its parallel test solenoid — the SV
        # bypass tees in upstream of the meter), so opening an SV is directly
        # observable on FT-10x for interlock-test evidence.
        self.q12_lpm = (flows["q_12"] + flows["q_sv1"]) * 60000.0
        self.q23_lpm = (flows["q_23"] + flows["q_sv2"]) * 60000.0
        self.q3r_lpm = (flows["q_3r"] + flows["q_sv3"]) * 60000.0

        # emulated safety flags (real ones are hardwired; mock reflects state).
        self.di_dryfire = 1 if self.h1 < p.low_level_trip else 0
        self.di_overflow = 1 if max(self.h1, self.h2, self.h3) > p.high_level_trip else 0
        self.di_heater_contactor = 1 if flows["Qh1"] > 0.0 else 0
        # P1/#6-re: the contactor follows the RUN COMMAND, not the flow — a real
        # contactor stays closed at sub-deadband speed with zero delivered flow,
        # and drops at zero speed. Keying on q_pump > 0 also never dropped under
        # the old phantom (and would flicker with the C1 gate's 0.0036 L/min
        # residue); the command is what the physical aux contact actually wired.
        self.di_pump_contactor = 1 if pump_frac > 0.0 else 0
        self.di_estop = 1      # NC chain healthy (never pressed in sim; 0 = pressed)

    def snapshot(self) -> dict:
        return {
            "h_cm": [round(self.h1 * 100, 2), round(self.h2 * 100, 2), round(self.h3 * 100, 2)],
            "T": [round(self.T1, 2), round(self.T2, 2), round(self.T3, 2)],
            "reservoir": {"h": round(self.h_res, 3), "T": round(self.T_res, 2)},
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
    proc.T_res = RESET_TEMP_C   # finite reservoir temp -> warm start (level persists)
    # Minor/#6: refresh DI flags + FT sensors for the SNAPPED state. They only
    # update in proc.step, which is skipped while the reset holds — so the next
    # episode's first observation carried the previous episode's dry-fire/
    # overflow flags and flow readings (and on IA2 the stale DI could latch ST
    # trips for the ~10-scan hold window). During the hold the actuators are
    # zeroed by the env, so flows are 0 and the contactor flags drop out.
    proc.q12_lpm = proc.q23_lpm = proc.q3r_lpm = 0.0
    proc.di_dryfire = 1 if proc.h1 < params.low_level_trip else 0
    proc.di_overflow = 1 if max(proc.h1, proc.h2, proc.h3) > params.high_level_trip else 0
    proc.di_heater_contactor = 0
    proc.di_pump_contactor = 0
    proc.di_estop = 1          # NC chain healthy


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
    (holding AND coil — the DO module's SV solenoids are written by the
    controller side and read back by the physics loop) start at zero. Note
    SimData `count` *repeats* the values list, so for a list of N distinct
    register values we leave count at its default (1).
    """
    by_table: dict[str, list[dict]] = {}
    for r in regs_on_slave:
        by_table.setdefault(r["table"], []).append(r)

    def block(table: str, datatype, is_read: bool) -> list:
        rs = sorted(by_table.get(table, []), key=lambda r: r["address"])
        zero = False if datatype == DataType.BITS else 0
        if not rs:
            return [SimData(0, values=[zero], datatype=datatype)]  # non-empty placeholder
        # gap-aware: cover the full [lo, hi] span; gaps (e.g. AO 16-bit value regs at
        # 1,3,5,7 with unused 0,2,4,6) stay zero, values placed by address offset.
        lo = rs[0]["address"]
        hi = max(r["address"] + int(r["count"]) - 1 for r in rs)
        vals: list = [zero] * (hi - lo + 1)
        for r in rs:
            off = r["address"] - lo
            enc = encode_channel(r, proc, params) if is_read else [zero] * int(r["count"])
            vals[off:off + len(enc)] = enc
        return [SimData(lo, values=vals, datatype=datatype)]  # count defaults to 1

    return (block("coil", DataType.BITS, False),
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
        lo = regs[0]["address"]
        hi = max(r["address"] + int(r["count"]) - 1 for r in regs)
        holding_plan[sid] = (lo, hi - lo + 1, regs)   # gap-aware: read the full span
    # Coil read plan (DO module test solenoids) — FC01, same union-into-regval.
    coil_plan = {}
    for sid, regs in layout.coil_by_slave.items():
        lo = regs[0]["address"]
        hi = regs[-1]["address"]
        coil_plan[sid] = (lo, hi - lo + 1, regs)

    while True:
        # --- read all holding (actuator + reset) blocks, union into regval ---
        regval: dict = {}
        for sid, (base, count, regs) in holding_plan.items():
            raw = await server.async_getValues(sid, 3, base, count)
            regval.update(_decode_holding(raw, regs))
        for sid, (base, count, regs) in coil_plan.items():
            bits = await server.async_getValues(sid, 1, base, count)
            for r, b in zip(regs, bits):
                regval[r["name"]] = bool(b)

        # --- episode reset (nonce value-change on reset_cmd, slave 03) ---
        reset_val = int(regval.get(RESET_CMD, 0))
        if reset_val != 0 and reset_val != prev_reset_val:
            apply_reset(proc, regval, layout, params)
            LOG.info("reset applied -> %s", proc.snapshot())
        if reset_val == 0:
            proc.step(regval.get(VFD_CMD, 0.0), regval.get(V12_CMD, 0.0),
                      regval.get(V23_CMD, 0.0), regval.get(V33_CMD, 0.0),
                      regval.get(E101_CMD, 0.0), dt, params,
                      sv=(regval.get(SV1_CMD, False), regval.get(SV2_CMD, False),
                          regval.get(SV3_CMD, False)))
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
            LOG.info("vfd=%5.1fHz v12=%5.1f v23=%5.1f v33=%5.1f e101=%5.1f "
                     "sv=%d%d%d  %s",
                     regval.get(VFD_CMD, 0.0), regval.get(V12_CMD, 0.0),
                     regval.get(V23_CMD, 0.0), regval.get(V33_CMD, 0.0),
                     regval.get(E101_CMD, 0.0),
                     1 if regval.get(SV1_CMD, False) else 0,
                     1 if regval.get(SV2_CMD, False) else 0,
                     1 if regval.get(SV3_CMD, False) else 0,
                     proc.snapshot())

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
