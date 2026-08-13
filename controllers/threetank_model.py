"""Heated serial-cascade plant model for the MPC controllers (AIO-Gym MPCAgent).

Mirrors `mock_cabinet.py`'s physics — pump recirculation into Tank1, valve-modulated
Torricelli cascade (V-12 -> V-23 -> V-33), and a first-law thermal balance with a
SINGLE heater in Tank1 (Tank2/Tank3 warm via downstream advection) — so the MPC's
internal predictor matches the simulated plant. Params come from `ia2_config.json`
(single source). MUST stay drift-free with mock_cabinet.py and nmpc_oracle.py
(shared cleanup deferred to Phase 5).

State layout (interleaved, like AIO-Gym's cascade): x = [h1, T1, h2, T2, h3, T3].
Actions: pumps=[p1], valves=[V-12, V-23, V-33], heaters=[E-101] (each 0..1).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

# Gravity is universal; geometry + valve coefficients come from the contract.
G = 9.81
CONFIG = Path(__file__).resolve().parents[1] / "ia2_config.json"


def _valve_flow(h_upstream: float, c_v: float, frac: float,
                gravity_drop: float = 0.0) -> float:
    """Valve flow (m^3/s) — aligned with xinji's model. MUST match mock_cabinet._valve_flow.
    q = frac × cv × √(upstream_level + gravity_drop). No 2g (absorbed into cv)."""
    head = h_upstream + gravity_drop
    if head <= 1e-9:
        return 0.0
    frac = max(0.0, min(frac, 1.0))
    return frac * c_v * math.sqrt(head)


class ThreeTankModel:
    """Numpy heated serial-cascade plant model implementing the MPCAgent interface."""

    scenario = "threetank"   # not cstr/hvac/heater -> MPCAgent uses the interleaved branch
    n = 3
    energy_scored = True     # AIO-Gym KPIScorer: score the excess-heater-energy KPI

    def __init__(self):
        with open(CONFIG) as fh:
            cfg = json.load(fh)
        p = cfg["process"]
        self.q_max = float(p["q_max_m3s"])
        self.h_max = float(p["h_max_m"])
        self.t_supply = float(p["t_supply_c"])
        self.a_tank = float(p["a_tank_m2"])
        self.c_v12 = float(p["c_v12"])
        self.c_v23 = float(p["c_v23"])
        self.c_v33 = float(p["c_v33"])
        self.gravity_drop = float(p.get("gravity_drop_m", 0.0))
        self.pump_static_head = float(p.get("pump_static_head_m", 1.7))
        self.pump_shutoff_head = float(p.get("pump_shutoff_head_m", 10.0))
        self.overflow_level = float(p.get("overflow_level_m", 0.46))
        self.cv_overflow = float(p.get("cv_overflow", 0.001))
        self.overflow_head_floor = float(p.get("overflow_head_floor", 1e-9))
        self.reservoir_base = float(p.get("reservoir_base_m2", 0.30))
        self._last_t_res = self.t_supply  # cached reservoir temp (updated in derivatives, used in ideal_power)
        self.cp = float(p["cp_j_per_kgk"])
        self.rho = float(p["rho_kg_per_m3"])
        self.q_heat_max = float(p["q_heat_max_w"])
        self.ua = float(p["ua_w_per_k"])
        self.t_ambient = float(p["t_ambient_c"])
        # default setpoints (from the contract) for AIOGymNativeEnv.
        ctrl = cfg["control"]
        self._hsp = {0: float(ctrl["setpoints_m"]["tank1_level"]),
                     1: float(ctrl["setpoints_m"]["tank2_level"]),
                     2: float(ctrl["setpoints_m"]["tank3_level"])}
        self._tsp = [float(ctrl["setpoints_c"]["tank1_temp"]),
                     float(ctrl["setpoints_c"]["tank2_temp"]),
                     float(ctrl["setpoints_c"]["tank3_temp"])]
        # CasADi symbolic params (nmpc_oracle) + AIO-Gym env reads t_cold/t_amb.
        self.p = {
            "q_max": self.q_max, "A_TANK": self.a_tank, "q_heat_max": self.q_heat_max,
            "ua": self.ua, "cp": self.cp, "rho": self.rho,
            "C_V12": self.c_v12, "C_V23": self.c_v23, "C_V33": self.c_v33,
            "gravity_drop": self.gravity_drop,
            "pump_static_head": self.pump_static_head,
            "pump_shutoff_head": self.pump_shutoff_head,
            "overflow_level": self.overflow_level,
            "cv_overflow": self.cv_overflow,
            "overflow_head_floor": self.overflow_head_floor,
            "reservoir_base": self.reservoir_base,
            "G": G, "t_supply": self.t_supply, "t_ambient": self.t_ambient,
            "t_cold": self.t_supply, "t_amb": self.t_ambient,
            "h_floor": 0.02, "h_max": self.h_max,
        }
        self.dt_micro = 0.05

    # ---- MPCAgent interface ----
    def actuator_counts(self):
        return (1, 3, 1)          # 1 pump (VFD), 3 valves (V-12, V-23, V-33), 1 heater (E-101)

    def initial_state(self):
        return [0.30, 25.0, 0.22, 25.0, 0.18, 25.0, 0.30, 25.0]  # +h_res, T_res (finite reservoir)

    def controlled_levels(self):
        return [0, 1, 2]          # pump->T1, V-12->T2, V-23->T3 (V-33 is the T3 drain MV)

    def levels_temps(self, x):
        return ([max(x[0], 0.0), max(x[2], 0.0), max(x[4], 0.0)],
                [x[1], x[3], x[5]])

    def derivatives(self, x, act, env):
        """ODE RHS [dh1, dT1, dh2, dT2, dh3, dT3] given state + action + env."""
        h1, T1, h2, T2, h3, T3 = x[0], x[1], x[2], x[3], x[4], x[5]
        h_res, T_res = x[6], x[7]  # finite reservoir (internal, unmeasured)
        self._last_t_res = T_res    # cache for ideal_power
        pumps, valves, heaters = act["pumps"], act["valves"], act["heaters"]
        t_cold = env.get("t_cold", self.t_supply)
        t_amb = env.get("t_amb", self.t_ambient)

        # hydraulics: pump recirc into Tank1 + unidirectional valve cascade to reservoir
        # pump curve (aligned with xinji): q = pump_flow_max × √((shutoff×u²−static)/(shutoff−static))
        _hm = self.pump_shutoff_head - self.pump_static_head
        _nh = (self.pump_shutoff_head * pumps[0] ** 2 - self.pump_static_head) / _hm
        q_pump = self.q_max * max(_nh, 0.0) ** 0.5               # P-101 pump curve
        q_12 = _valve_flow(h1, self.c_v12, valves[0], self.gravity_drop)
        q_23 = _valve_flow(h2, self.c_v23, valves[1], self.gravity_drop)
        q_3r = _valve_flow(h3, self.c_v33, valves[2], self.gravity_drop)
        # hydraulic overflow (aligned with xinji): spills when level > overflow_level
        def _ovf(h):
            head = h - self.overflow_level
            return self.cv_overflow * max(head, self.overflow_head_floor) ** 0.5 if head > 0.0 else 0.0
        dh1 = (q_pump - q_12 - _ovf(h1)) / self.a_tank
        dh2 = (q_12 - q_23 - _ovf(h2)) / self.a_tank
        dh3 = (q_23 - q_3r - _ovf(h3)) / self.a_tank

        # thermal: ONE heater in Tank1; chain advection carries heat downstream
        adv1 = q_pump * (T_res - T1)      # finite reservoir: pump draws at T_res      # recirc returns reservoir-temp water to Tank1
        adv2 = q_12 * (T1 - T2)            # hot Tank1 outflow -> Tank2
        adv3 = q_23 * (T2 - T3)            # Tank2 outflow -> Tank3 (Tank3 loses via q_3r: outflow drops out)
        dT1 = self._dT(T1, h1, heaters[0] * self.q_heat_max, adv1, t_amb)
        dT2 = self._dT(T2, h2, 0.0, adv2, t_amb)
        dT3 = self._dT(T3, h3, 0.0, adv3, t_amb)

        # finite reservoir dynamics (internal, unmeasured — same as mock_cabinet)
        dh_res = (q_3r - q_pump) / self.reservoir_base
        adv_res = q_3r * (T3 - T_res)
        dT_res = self._dT(T_res, h_res, 0.0, adv_res, t_amb, self.reservoir_base)
        return [dh1, dT1, dh2, dT2, dh3, dT3, dh_res, dT_res]

    def _dT(self, T, h, q_heat, adv, t_amb, area=None):
        area = area if area is not None else self.a_tank
        h = max(h, 0.02)
        m_cp = self.rho * area * h * self.cp
        q_loss = self.ua * (T - t_amb)
        return (q_heat - q_loss) / m_cp + adv / (area * h)

    # ---- AIO-Gym env / KPIScorer interface (KPI + economic reward) ----
    def default_setpoints(self):
        return dict(self._hsp), list(self._tsp)

    @property
    def height_max(self):
        return [self.h_max, self.h_max, self.h_max]

    def heater_power(self, act):
        return act["heaters"][0] * self.q_heat_max

    def ideal_power(self, levels, temps, t_sp, env, act):
        """Thermodynamic floor for the excess-energy KPI: power to warm the cold
        recirc inflow into Tank1 to t_sp[0] and cover Tank1's heat loss. Tank2/Tank3
        are warmed only by advection (no direct heater), so they are not in the floor."""
        # pump curve (must match derivatives — was linear, which inflated the floor)
        _hm = self.pump_shutoff_head - self.pump_static_head
        _nh = (self.pump_shutoff_head * act["pumps"][0] ** 2 - self.pump_static_head) / _hm
        q_pump = self.q_max * max(_nh, 0.0) ** 0.5
        rho_cp = self.rho * self.cp
        t_amb = env["t_amb"]
        t_inflow = self._last_t_res  # finite reservoir temp (cached from derivatives)
        return max(0.0, rho_cp * q_pump * (t_sp[0] - t_inflow) + self.ua * (t_sp[0] - t_amb))

    def clamp_state(self, x):
        return x
