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

# Strictly-positive floor for every sqrt argument (B2/#6): keeps d/dx sqrt(x)
# finite for the CasADi/NMPC Jacobian. 1e-9 matches the pre-consolidation oracle.
_SQRT_FLOOR = 1e-9
# M1/#6: availability ramp width (m). Flows scale by min(1, level/eps) so a dry
# tank cannot deliver water (the old head = level + gravity_drop kept ~47 L/min
# flowing at level 0) and the pump cannot draw from an empty reservoir.
_LEVEL_EPS = 0.02
# P1/#6-re: pump gate ramp width in net head (dimensionless, nh spans [-0.2, 1]).
# 0.01 spans duty u* -> u*+0.010 (~1.0% of the action box; dnh/du = 20u/8.3
# ≈ 0.994 at u*) where the true curve delivers <= 6.7 L/min on a 67 L/min
# pump. Narrower (5e-4) left curvature ~1/eps that stalled IPOPT on high-level
# starts; wider is curve-fidelity loss for nothing. See compute_flows.
# (#6-re: the old "~0.3% of the action box" note was wrong — re-derived.)
_PUMP_GATE_EPS = 0.01
# P1/#6-re: the weir gates get their own, much narrower ramp. The tank-drain
# _LEVEL_EPS (0.02 m) is wider than the entire reachable weir band
# (0.46 -> 0.475 = 0.015 m), so relief flow was attenuated 25-87% everywhere
# the weir can physically operate.
_OVF_EPS = 0.002


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
        # On/off interlock-test solenoids (SV-1..3, parallel to V-12/V-23/V-33).
        # getattr: the MPC model may predate the field — and with sv_fracs
        # defaulting closed the coefficient is multiplied by 0 anyway.
        "cv_sv": getattr(m, "cv_sv", 0.00143),
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


def compute_flows(x, pump_frac, valve_fracs, heater_frac, p, ops,
                  sv_fracs=(0.0, 0.0, 0.0)):
    """Single source of the hydraulic + heater flow equations.

    Returns the intermediate flows (m^3/s) and heater power (W). Consumed by both
    dynamics() (for the derivatives) and mock_cabinet.py (for the published FT
    sensors + the emulated contactor DI flags) — so the mock's published flows
    match the model's internal flows exactly. No drift.

    pump curve (xinji): q = q_max × √((shutoff×u² − static)/(shutoff − static))
    valve flow (xinji): q = cv × frac × √(level + gravity_drop)
    overflow:           q = cv_ovf × √(max(level − overflow_level, floor))
    test solenoid:      q = cv_sv × sv ∈ {0,1} × √(level + gravity_drop)
                        (SV-i is the on/off interlock-test bypass PARALLEL to
                        valve i; sv_fracs defaults closed so MPC/NMPC — which
                        never actuate test hardware — get bit-identical
                        dynamics with the default argument)
    """
    _max = ops["max"]
    _sqrt = ops["sqrt"]
    h1, h2, h3 = x[0], x[2], x[4]
    cvs = p["cv_valves"]
    g_drop = p["gravity_drop"]

    def _smax(a, b, eps=1e-6):
        """Smooth max — B2/#6: the quadratic pump curve hits its clamp at
        u* = sqrt(static/shutoff) ≈ 0.41, inside the [0,1] action box, and the
        hard fmax kink there stalls IPOPT. This form is C1-smooth, equals max
        to within sqrt(eps)/2 (~5e-4) at the kink and ~eps/(2|a-b|) away from
        it, and is IDENTICAL in numpy and CasADi (same formula), so the shared-
        physics no-drift property is preserved."""
        return 0.5 * (a + b + _sqrt((a - b) * (a - b) + eps))

    def _smin(a, b, eps=1e-6):
        return 0.5 * (a + b - _sqrt((a - b) * (a - b) + eps))

    def _avail(h, h_eps=_LEVEL_EPS):
        """M1/#6 availability factor min(1, max(h,0)/h_eps), smoothed. A dry tank
        must not deliver water — the bare orifice head (level + gravity_drop)
        kept ~47 L/min flowing at level = 0, and the pump kept feeding Tank1 from
        an empty reservoir (the mock's clamp froze h_res at 0 while mass appeared
        from nothing). Exactly 0 below the ramp, 1 (to ~1e-9) above it."""
        return _smin(1.0, _max(h, 0.0) / h_eps)

    h_res = x[6]
    hm = p["pump_shutoff_head"] - p["pump_static_head"]
    nh = (p["pump_shutoff_head"] * pump_frac * pump_frac - p["pump_static_head"]) / hm
    # B2/#6: the sqrt argument must stay strictly positive — d/dx sqrt(x) is
    # infinite at 0, which NaNs the NMPC Jacobian over the whole u_pump <= 0.41
    # band. _smax(nh, 0) is smooth AND bounded below by sqrt(eps)/2 > 0.
    # P1/#6-re: but that floor is itself a phantom — _smax(nh, 0) is strictly
    # positive for nh < 0, so the pump delivered ~0.074 L/min at duty 0.0 (up to
    # 1.5 L/min across the deadband): an idle plant slowly filled Tank1 (overflow
    # trip after ~3 h), the NMPC treated the phantom band as real authority
    # (nominal pump 0.365, inside the deadband), and the pump-contactor DI
    # (formerly keyed on q_pump > 0) never dropped. Fix: multiply by a smooth
    # availability ramp on net head — a pure _smin/_smax composition, C1
    # EVERYWHERE. This matters: any exact-zero-below formulation (a hard clamp,
    # or a constant-floor subtraction) has a slope jump 0 -> finite at nh = 0,
    # and the NMPC's drain-regime optimum sits exactly on that corner — IPOPT
    # cannot close the RK4 defects across it (measured: constraint-violation
    # flutter at ~4e-4 out to max-iter, 40 s per stalled solve, on high-level
    # starts). The C1 gate's price is a 9e-6 L/min residue at duty 0.0 (1.4e-7
    # of q_max; idle drift 0.0165 cm/day — still ~8300x below the 0.074 L/min
    # phantom it kills; #6-re re-measured: the old "<0.002 cm/day" was 8x low)
    # — so the mock's contactor DI is keyed on the pump COMMAND instead, which
    # is the physical contactor anyway (it stays closed at sub-deadband speed
    # with zero
    # flow). Above the 0.01-wide ramp the curve is unbiased to ~1e-7 rel, and no
    # constant-floor subtraction bias exists.
    # M1/#6: gated on the reservoir level — no pumping from an empty reservoir.
    pump_gate = _smin(1.0, _smax(nh, 0.0) / _PUMP_GATE_EPS)
    q_pump = p["q_max"] * _sqrt(_smax(nh, 0.0)) * pump_gate * _avail(h_res)

    q_12 = cvs[0] * valve_fracs[0] * _sqrt(_max(h1 + g_drop, _SQRT_FLOOR)) * _avail(h1)
    q_23 = cvs[1] * valve_fracs[1] * _sqrt(_max(h2 + g_drop, _SQRT_FLOOR)) * _avail(h2)
    q_3r = cvs[2] * valve_fracs[2] * _sqrt(_max(h3 + g_drop, _SQRT_FLOOR)) * _avail(h3)

    # SV-1..3: on/off interlock-test solenoids, each PARALLEL to its modulating
    # valve (same line, full-port — same Torricelli form, binary frac). Open one
    # and you get full-bore flow on that path regardless of valve position: the
    # scripted level transient for SAT interlock testing (LSH/LSL trips).
    cv_sv = p.get("cv_sv", 0.00143)
    q_sv1 = cv_sv * sv_fracs[0] * _sqrt(_max(h1 + g_drop, _SQRT_FLOOR)) * _avail(h1)
    q_sv2 = cv_sv * sv_fracs[1] * _sqrt(_max(h2 + g_drop, _SQRT_FLOOR)) * _avail(h2)
    q_sv3 = cv_sv * sv_fracs[2] * _sqrt(_max(h3 + g_drop, _SQRT_FLOOR)) * _avail(h3)

    cv_ovf = p["cv_overflow"]
    ovf_level = p["overflow_level"]
    # M1/#6: the availability gate also kills the sub-weir leak — the old
    # sqrt(max(level - overflow_level, 1e-9)) bled ~8 L/day per tank with the
    # level below the weir. _smax keeps the sqrt argument (and its CasADi
    # derivative) strictly positive; the gate makes the flow exactly 0 below.
    # P1/#6-re: the weir gates use _OVF_EPS (0.002 m), NOT the tank-drain
    # _LEVEL_EPS — reusing the drain ramp width attenuated legitimate relief
    # 25-87% across the whole 0.015 m weir band.
    ovf1 = cv_ovf * _sqrt(_smax(h1 - ovf_level, _SQRT_FLOOR)) * _avail(h1 - ovf_level, _OVF_EPS)
    ovf2 = cv_ovf * _sqrt(_smax(h2 - ovf_level, _SQRT_FLOOR)) * _avail(h2 - ovf_level, _OVF_EPS)
    ovf3 = cv_ovf * _sqrt(_smax(h3 - ovf_level, _SQRT_FLOOR)) * _avail(h3 - ovf_level, _OVF_EPS)

    return {
        "q_pump": q_pump, "q_12": q_12, "q_23": q_23, "q_3r": q_3r,
        "q_sv1": q_sv1, "q_sv2": q_sv2, "q_sv3": q_sv3,
        "ovf1": ovf1, "ovf2": ovf2, "ovf3": ovf3,
        "Qh1": heater_frac * p["q_heat_max"],
    }


def dynamics(x, pump_frac, valve_fracs, heater_frac, p, ops,
             sv_fracs=(0.0, 0.0, 0.0)):
    """8-dim ODE RHS [dh1, dT1, dh2, dT2, dh3, dT3, dh_res, dT_res].

    Aligned with xinji's AIO-Gym v0.2 model: orifice valve flow
    (cv × √(level + gravity_drop)), quadratic pump curve, hydraulic overflow,
    well-mixed thermal, finite reservoir. sv_fracs are the on/off interlock-test
    solenoids (parallel to V-12/V-23/V-33); default closed.

    x = [h1, T1, h2, T2, h3, T3, h_res, T_res]
    Returns a list (numpy) or MX vector (CasADi — caller wraps in vertcat).
    """
    _max = ops["max"]
    f = compute_flows(x, pump_frac, valve_fracs, heater_frac, p, ops, sv_fracs)

    h1, T1, h2, T2, h3, T3 = x[0], x[1], x[2], x[3], x[4], x[5]
    h_res, T_res = x[6], x[7]

    area = p["area"]
    rho_cp = p["rho"] * p["cp"]
    h_floor = p["h_floor"]

    # path totals (modulating valve + parallel test solenoid): both legs move
    # the same water between the same pair of tanks, so every balance below —
    # mass AND advection — must see the SUM.
    q12_t = f["q_12"] + f["q_sv1"]
    q23_t = f["q_23"] + f["q_sv2"]
    q3r_t = f["q_3r"] + f["q_sv3"]

    # ---- mass balance ----
    dh1 = (f["q_pump"] - q12_t - f["ovf1"]) / area
    dh2 = (q12_t - q23_t - f["ovf2"]) / area
    dh3 = (q23_t - q3r_t - f["ovf3"]) / area
    # M1/#6: overflow spills RETURN to the reservoir (the rig's weirs drain into
    # it) — the old balance subtracted ovf from the tanks and dropped it, so loop
    # inventory decayed whenever a tank rode the weir.
    dh_res = (q3r_t + f["ovf1"] + f["ovf2"] + f["ovf3"] - f["q_pump"]) / p["reservoir_base"]

    # ---- thermal: well-mixed tank energy balance ----
    # dT = (Q_heat − Q_loss) / (rho × area × h × cp) + adv / (area × h)
    adv1 = f["q_pump"] * (T_res - T1)
    adv2 = q12_t * (T1 - T2)
    adv3 = q23_t * (T2 - T3)

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

    adv_res = (q3r_t * (T3 - T_res) + f["ovf1"] * (T1 - T_res)
               + f["ovf2"] * (T2 - T_res) + f["ovf3"] * (T3 - T_res))
    dT_res = (0.0 - p["ua"] * (T_res - t_amb)) / m_cp_res + adv_res / (p["reservoir_base"] * h_rs)

    return [dh1, dT1, dh2, dT2, dh3, dT3, dh_res, dT_res]


__all__ = ["NUMERIC_OPS", "casadi_ops", "build_params", "compute_flows", "dynamics"]
