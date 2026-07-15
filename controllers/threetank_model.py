"""3-tank plant model for the MPC controllers (AIO-Gym MPCAgent interface).

Mirrors `mock_cabinet.py`'s physics — Torricelli inter-tank hydraulics + the
first-law thermal energy balance (level-coupled mass, well-mixed-tank advection)
— so the MPC's internal predictor matches the simulated plant. Params come from
`ia2_config.json` (single source); only the ODE right-hand sides are mirrored
here. (A future cleanup could share one `derivatives()` between this and
mock_cabinet to eliminate even equation drift.)

State layout (interleaved, like AIO-Gym's cascade): x = [h1, T1, h2, T2, h3, T3].
Actions: pumps=[p1, p2], valves=[], heaters=[h1, h2, h3] (each 0..1).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

# Internal tuning constants (match mock_cabinet.py exactly).
G = 9.81
A_TANK = 0.0154      # tank cross-section, m^2
S_PIPE = 5.0e-5      # connecting-pipe cross-section, m^2
A1 = A2 = A3 = 0.5   # outflow coefficients (T1<->T2, T2->drain, T3<->T2)

CONFIG = Path(__file__).resolve().parents[1] / "ia2_config.json"


def _flow(h_from: float, h_to: float, coeff: float) -> float:
    dh = h_from - h_to
    if abs(dh) < 1e-9:
        return 0.0
    return coeff * S_PIPE * math.copysign(math.sqrt(2.0 * G * abs(dh)), dh)


def _pos(x: float) -> float:
    return x if x > 0.0 else 0.0


def _neg(x: float) -> float:
    return -x if x < 0.0 else 0.0


class ThreeTankModel:
    """Numpy 3-tank plant model implementing the MPCAgent model interface."""

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
        # params dict + micro-step for the CasADi symbolic dynamics (nmpc_oracle)
        # and the AIO-Gym env (which reads p["t_cold"], p["t_amb"]).
        self.p = {
            "q_max": self.q_max, "A_TANK": A_TANK, "q_heat_max": self.q_heat_max,
            "ua": self.ua, "cp": self.cp, "rho": self.rho, "S_PIPE": S_PIPE,
            "A1": A1, "A2": A2, "A3": A3, "G": G, "t_supply": self.t_supply,
            "t_ambient": self.t_ambient, "t_cold": self.t_supply, "t_amb": self.t_ambient,
            "h_floor": 0.02, "h_max": self.h_max,
        }
        self.dt_micro = 0.05

    # ---- MPCAgent interface ----
    def actuator_counts(self):
        return (2, 0, 3)          # 2 pumps, 0 valves, 3 heaters

    def initial_state(self):
        return [0.30, 24.0, 0.18, 24.0, 0.24, 24.0]

    def controlled_levels(self):
        return [0, 2]             # tank1, tank3 (tank2 is the middle/downstream)

    def levels_temps(self, x):
        return ([max(x[0], 0.0), max(x[2], 0.0), max(x[4], 0.0)],
                [x[1], x[3], x[5]])

    def derivatives(self, x, act, env):
        """ODE RHS [dh1, dT1, dh2, dT2, dh3, dT3] given state + action + env."""
        h1, T1, h2, T2, h3, T3 = x
        pumps, heaters = act["pumps"], act["heaters"]
        t_cold = env.get("t_cold", self.t_supply)
        t_amb = env.get("t_amb", self.t_ambient)

        q1 = pumps[0] * self.q_max
        q2 = pumps[1] * self.q_max
        q_12 = _flow(h1, h2, A1)
        q_32 = _flow(h3, h2, A3)
        q_drain = _flow(h2, 0.0, A2)
        dh1 = (q1 - q_12) / A_TANK
        dh3 = (q2 - q_32) / A_TANK
        dh2 = (q_12 + q_32 - q_drain) / A_TANK

        f12, b12 = _pos(q_12), _neg(q_12)
        f32, b32 = _pos(q_32), _neg(q_32)
        adv1 = q1 * (t_cold - T1) + b12 * (T2 - T1)
        adv2 = f12 * (T1 - T2) + f32 * (T3 - T2)
        adv3 = q2 * (t_cold - T3) + b32 * (T2 - T3)

        dT1 = self._dT(T1, h1, heaters[0] * self.q_heat_max, adv1, t_amb)
        dT2 = self._dT(T2, h2, heaters[1] * self.q_heat_max, adv2, t_amb)
        dT3 = self._dT(T3, h3, heaters[2] * self.q_heat_max, adv3, t_amb)
        return [dh1, dT1, dh2, dT2, dh3, dT3]

    def _dT(self, T, h, q_heat, adv, t_amb):
        h = max(h, 0.02)
        m_cp = self.rho * A_TANK * h * self.cp
        q_loss = self.ua * (T - t_amb)
        return (q_heat - q_loss) / m_cp + adv / (A_TANK * h)

    # ---- AIO-Gym env / KPIScorer interface (KPI + economic reward) ----
    def default_setpoints(self):
        return dict(self._hsp), list(self._tsp)

    @property
    def height_max(self):
        return [self.h_max, self.h_max, self.h_max]

    def heater_power(self, act):
        return sum(u * self.q_heat_max for u in act["heaters"])

    def ideal_power(self, levels, temps, t_sp, env, act):
        """Steady-state heater power (W) to hold t_sp: warm the cold inflow + cover
        heat loss. The thermodynamic floor for the excess-energy KPI."""
        q1 = act["pumps"][0] * self.q_max
        q2 = act["pumps"][1] * self.q_max
        rho_cp = self.rho * self.cp
        t_amb, t_cold = env["t_amb"], env["t_cold"]
        tot = 0.0
        tot += max(0.0, rho_cp * q1 * (t_sp[0] - t_cold) + self.ua * (t_sp[0] - t_amb))
        tot += max(0.0, self.ua * (t_sp[1] - t_amb))            # middle tank: inflow ~ at t_sp
        tot += max(0.0, rho_cp * q2 * (t_sp[2] - t_cold) + self.ua * (t_sp[2] - t_amb))
        return tot

    def clamp_state(self, x):
        return x
