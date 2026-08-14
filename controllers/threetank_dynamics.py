"""Shared 8-dim ODE right-hand side for the heated serial-cascade plant.

One implementation of the valve flow, pump curve, overflow, thermal, and finite
reservoir dynamics — imported by mock_cabinet.py, threetank_model.py, and
nmpc_oracle.py. Eliminates equation-drift risk: change the physics here and all
three consumers see the update.

Uses an `ops` dict abstraction for numpy vs CasADi backends:
    ops = NUMERIC_OPS               # for mock + model (numpy)
    ops = casadi_ops(ca)             # for oracle (CasADi symbolic)

State layout: x = [h1, T1, h2, T2, h3, T3, h_res, T_res] (8-dim).
Action: pump_frac [0,1], valve_fracs [v12, v23, v33] each [0,1], heater_frac [0,1].
Params: a flat dict (see build_params below) with all physics constants.
"""
from __future__ import annotations

import math


# ---- ops backends ----
NUMERIC_OPS = {
    "max": lambda a, b: max(a, b),
    "sqrt": lambda x: math.sqrt(x) if x > 0 else 0.0,
    "fmax": lambda a, b: max(a, b),
}


def casadi_ops(ca):
    """Build an ops dict for CasADi symbolic computation."""
    return {
        "max": ca.fmax,
        "sqrt": ca.sqrt,
        "fmax": ca.fmax,
    }


def build_params(model_or_physics):
    """Extract the flat params dict from a ThreeTankModel or PhysicsParams.

    Both sources carry the same physics values (from ia2_config.json); this
    normalises them into one dict for the shared dynamics.
    """
    m = model_or_physics
    return {
        "q_max": m.q_max,
        "area": m.a_tank,
        "cv_valves": [m.c_v12, m.c_v23, m.c_v33],
        "gravity_drop": m.gravity_drop,
        "pump_static_head": m.pump_static_head,
        "pump_shutoff_head": m.pump_shutoff_head,
        "overflow_level": m.overflow_level,
        "cv_overflow": m.cv_overflow,
        "overflow_head_floor": m.overflow_head_floor,
        "ua": m.ua,
        "cp": m.cp,
        "rho": m.rho,
        "q_heat_max": m.q_heat_max,
        "reservoir_base": m.reservoir_base,
        "h_floor": 0.02,
        "t_ambient": getattr(m, "t_ambient", 25.0),
    }


def compute_flows(x, pump_frac, valve_fracs, heater_frac, p, ops):
    """Single source of the hydraulic + heater flow equations.

    Returns the intermediate flows (m^3/s) and heater power (W). Consumed by both
    dynamics() (for the derivatives) and mock_cabinet.py (for the published FT
    sensors + the emulated contactor DI flags) — so the mock's published flows
    match the model's internal flows exactly. No drift.

    pump curve (xinji): q = q_max × √((shutoff×u² − static)/(shutoff − static))
    valve flow (xinji): q = cv × frac × √(level + gravity_drop)
    overflow:           q = cv_ovf × √(max(level − overflow_level, floor))
    """
    _max = ops["max"]
    _sqrt = ops["sqrt"]
    h1, h2, h3 = x[0], x[2], x[4]
    cvs = p["cv_valves"]
    g_drop = p["gravity_drop"]

    hm = p["pump_shutoff_head"] - p["pump_static_head"]
    nh = (p["pump_shutoff_head"] * pump_frac * pump_frac - p["pump_static_head"]) / hm
    q_pump = p["q_max"] * _sqrt(_max(nh, 0.0))

    q_12 = cvs[0] * valve_fracs[0] * _sqrt(_max(h1 + g_drop, 0.0))
    q_23 = cvs[1] * valve_fracs[1] * _sqrt(_max(h2 + g_drop, 0.0))
    q_3r = cvs[2] * valve_fracs[2] * _sqrt(_max(h3 + g_drop, 0.0))

    cv_ovf = p["cv_overflow"]
    ovf_level = p["overflow_level"]
    ovf_floor = p["overflow_head_floor"]
    ovf1 = cv_ovf * _sqrt(_max(h1 - ovf_level, ovf_floor))
    ovf2 = cv_ovf * _sqrt(_max(h2 - ovf_level, ovf_floor))
    ovf3 = cv_ovf * _sqrt(_max(h3 - ovf_level, ovf_floor))

    return {
        "q_pump": q_pump, "q_12": q_12, "q_23": q_23, "q_3r": q_3r,
        "ovf1": ovf1, "ovf2": ovf2, "ovf3": ovf3,
        "Qh1": heater_frac * p["q_heat_max"],
    }


def dynamics(x, pump_frac, valve_fracs, heater_frac, p, ops):
    """8-dim ODE RHS [dh1, dT1, dh2, dT2, dh3, dT3, dh_res, dT_res].

    Aligned with xinji's AIO-Gym v0.2 model: orifice valve flow
    (cv × √(level + gravity_drop)), quadratic pump curve, hydraulic overflow,
    well-mixed thermal, finite reservoir.

    x = [h1, T1, h2, T2, h3, T3, h_res, T_res]
    Returns a list (numpy) or MX vector (CasADi — caller wraps in vertcat).
    """
    _max = ops["max"]
    f = compute_flows(x, pump_frac, valve_fracs, heater_frac, p, ops)

    h1, T1, h2, T2, h3, T3 = x[0], x[1], x[2], x[3], x[4], x[5]
    h_res, T_res = x[6], x[7]

    area = p["area"]
    rho_cp = p["rho"] * p["cp"]
    h_floor = p["h_floor"]

    # ---- mass balance ----
    dh1 = (f["q_pump"] - f["q_12"] - f["ovf1"]) / area
    dh2 = (f["q_12"] - f["q_23"] - f["ovf2"]) / area
    dh3 = (f["q_23"] - f["q_3r"] - f["ovf3"]) / area
    dh_res = (f["q_3r"] - f["q_pump"]) / p["reservoir_base"]

    # ---- thermal: well-mixed tank energy balance ----
    # dT = (Q_heat − Q_loss) / (rho × area × h × cp) + adv / (area × h)
    adv1 = f["q_pump"] * (T_res - T1)
    adv2 = f["q_12"] * (T1 - T2)
    adv3 = f["q_23"] * (T2 - T3)

    h1s = _max(h1, h_floor)
    h2s = _max(h2, h_floor)
    h3s = _max(h3, h_floor)
    h_rs = _max(h_res, h_floor)

    m_cp1 = rho_cp * area * h1s
    m_cp2 = rho_cp * area * h2s
    m_cp3 = rho_cp * area * h3s
    m_cp_res = rho_cp * p["reservoir_base"] * h_rs

    t_amb = p.get("t_ambient", 25.0)

    dT1 = (f["Qh1"] - p["ua"] * (T1 - t_amb)) / m_cp1 + adv1 / (area * h1s)
    dT2 = (0.0 - p["ua"] * (T2 - t_amb)) / m_cp2 + adv2 / (area * h2s)
    dT3 = (0.0 - p["ua"] * (T3 - t_amb)) / m_cp3 + adv3 / (area * h3s)

    adv_res = f["q_3r"] * (T3 - T_res)
    dT_res = (0.0 - p["ua"] * (T_res - t_amb)) / m_cp_res + adv_res / (p["reservoir_base"] * h_rs)

    return [dh1, dT1, dh2, dT2, dh3, dT3, dh_res, dT_res]


__all__ = ["NUMERIC_OPS", "casadi_ops", "build_params", "compute_flows", "dynamics"]
