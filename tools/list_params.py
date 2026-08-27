#!/usr/bin/env python3
"""List the tank-system physical parameters, AIO-Gym 'list parameters' style.

`aiogym list parameters --scenario three_tank` (xinji's repo) prints a
NAME / VALUE / UNIT table of scenario parameters. This is the sandbox's
counterpart: it reads the single contract (ia2_config.json -> process) —
the same numbers mock_cabinet.py, threetank_model.py and nmpc_oracle.py
simulate — and prints them in that table shape, using xinji-compatible
names where the plants correspond. Diffing this table against xinji's (or
against the final measured rig parameters) is then a straight side-by-side,
no AIO-Gym clone required.

Rows marked `*` are physics-derived estimates pending bench SAT measurement
(config `process._placeholders`) — exactly the values to re-check once the
physical parameters are final.

Usage:
    python3 tools/list_params.py            # table (core + extensions)
    python3 tools/list_params.py --json     # flat dict, for tooling/diffing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Core parameters — one row per xinji `three_tank` name. `key` is the
# ia2_config.json process key (missing key = loud error, so renames can't
# silently drop a row); `vec3`/`valves` expand scalars to tank/valve lists
# the way xinji's table shows them.
CORE = [
    # (xinji name,          config key,             shape,      unit)
    ("area",                "a_tank_m2",            "tank",     "m^2"),
    ("cv_overflow",         "cv_overflow",          "tank",     "m^(5/2)/s"),
    ("cv_valves",           ("c_v12", "c_v23", "c_v33"), "keys", "m^(5/2)/s"),
    ("gravity_drop",        "gravity_drop_m",       "tank",     "m"),
    ("height_max",          "h_max_m",              "tank",     "m"),
    ("high_level_trip",     "high_level_trip_m",    "tank",     "m"),
    ("level_sensor_range",  "h_max_m",              "tank",     "m"),
    ("nominal_level",       "nominal_level_m",      "scalar",   "m"),
    ("overflow_head_floor", "overflow_head_floor",  "scalar",   "m"),
    ("overflow_level",      "overflow_level_m",     "tank",     "m"),
    ("pump_flow_max",       "q_max_m3s",            "scalar",   "m^3/s"),
    ("pump_power_max",      "pump_power_max_w",     "scalar",   "W"),
    ("pump_shutoff_head",   "pump_shutoff_head_m",  "scalar",   "m"),
    ("pump_static_head",    "pump_static_head_m",   "scalar",   "m"),
]
LEVEL_SENSOR_NOTE = "LT spans the full tank (h_max_m)"

# Heated-cascade additions — parameters not present in xinji's three_tank
# (SV bypasses, heater train, finite reservoir, extra trips).
EXTENSIONS = [
    ("cv_solenoids",   "c_sv",              "scalar", "m^(5/2)/s"),
    ("heat_loss_ua",   "ua_w_per_k",        "scalar", "W/K"),
    ("heater_power_max", "q_heat_max_w",    "scalar", "W"),
    ("low_level_trip", "low_level_trip_m",  "tank",   "m"),
    ("pump_vfd_max",   "vfd_max_hz",        "scalar", "Hz"),
    ("reservoir_area", "reservoir_base_m2", "scalar", "m^2"),
    ("reservoir_height", "reservoir_height_m", "scalar", "m"),
    ("t_ambient",      "t_ambient_c",       "scalar", "degC"),
    ("t_supply",       "t_supply_c",        "scalar", "degC"),
    ("temperature_trip", "temperature_trip_c", "scalar", "degC"),
    ("water_cp",       "cp_j_per_kgk",      "scalar", "J/(kg*K)"),
    ("water_rho",      "rho_kg_per_m3",     "scalar", "kg/m^3"),
]


def _fmt(v: float) -> str:
    return repr(float(v))


def _value(proc: dict, key, shape: str):
    """Resolve a row's value + the config keys it came from (for the
    SAT-pending marker). tank/valve shapes expand scalars to 3-lists."""
    if shape == "keys":                       # per-valve tuple, e.g. cv_valves
        return [float(proc[k]) for k in key], list(key)
    if shape == "tank":                       # one scalar -> [t, t, t]
        return [float(proc[key])] * 3, [key]
    return float(proc[key]), [key]


def build_rows(proc: dict, table: list) -> list[dict]:
    rows = []
    for name, key, shape, unit in table:
        value, keys = _value(proc, key, shape)
        rows.append({"name": name, "value": value, "unit": unit, "keys": keys})
    return rows


def render(rows: list[dict], placeholders: set[str], notes: dict[str, str]) -> str:
    def cell(v):
        if isinstance(v, list):
            return "[" + ", ".join(_fmt(x) for x in v) + "]"
        return _fmt(v)

    marked = [r | {"star": any(k in placeholders for k in r["keys"])} for r in rows]
    w_name = max(len(r["name"]) for r in marked)
    w_val = max(len(cell(r["value"])) for r in marked)
    # last column stays unpadded (no trailing spaces) — the * / note trail it
    lines = [f"{'NAME':<{w_name}}  {'VALUE':<{w_val}}  UNIT"]
    for r in sorted(marked, key=lambda r: r["name"]):
        star = " *" if r["star"] else ""
        note = f"  # {notes[r['name']]}" if r["name"] in notes else ""
        lines.append((f"{r['name']:<{w_name}}  {cell(r['value']):<{w_val}}  "
                      f"{r['unit']}{star}{note}").rstrip())
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "ia2_config.json"),
                    help="path to ia2_config.json (default: the repo contract)")
    ap.add_argument("--json", action="store_true",
                    help="print a flat JSON dict instead of a table")
    args = ap.parse_args(argv)

    proc = json.load(open(args.config))["process"]
    placeholders = set(proc.get("_placeholders", []))
    core = build_rows(proc, CORE)
    ext = build_rows(proc, EXTENSIONS)

    if args.json:
        print(json.dumps({
            "parameters": {r["name"]: r["value"] for r in sorted(core, key=lambda r: r["name"])},
            "extensions": {r["name"]: r["value"] for r in sorted(ext, key=lambda r: r["name"])},
            "sat_pending": sorted(placeholders),
        }, indent=1))
        return 0

    print("tank-system parameters (scenario 'threetank', heated cascade)")
    print(render(core, placeholders, {"level_sensor_range": LEVEL_SENSOR_NOTE}))
    print("\nheated-cascade additions (not present in xinji's three_tank)")
    print(render(ext, placeholders, {}))
    if placeholders:
        print("\n*  physics-derived estimate — pending bench SAT measurement "
              "(config process._placeholders)")
    print("\nsource: ia2_config.json -> process (the single contract; "
          "mock_cabinet / threetank_model / nmpc_oracle all simulate these)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
