"""Heated serial-cascade plant model for the MPC controllers (AIO-Gym MPCAgent).

Physics DELEGATES to the shared `controllers/threetank_dynamics.py` (one
implementation serving the mock, this model, and the NMPC oracle — no drift).
Params come from `ia2_config.json` (single source).

State layout (interleaved + finite reservoir): x = [h1, T1, h2, T2, h3, T3, h_res, T_res].
Actions: pumps=[p1], valves=[V-12, V-23, V-33], heaters=[E-101] (each 0..1).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

# Gravity is universal; geometry + valve coefficients come from the contract.
G = 9.81
CONFIG = Path(__file__).resolve().parents[1] / "ia2_config.json"


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
        # on/off interlock-test solenoids (SV-1..3): the controller never
        # actuates them (sv_fracs stays closed) — present only so the shared
        # dynamics' build_params reads the contract value from this model too.
        self.cv_sv = float(p.get("c_sv", 0.00143))
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
        """ODE RHS (8-dim) given state + action + env. Delegates to the shared
        threetank_dynamics module so the model, mock, and oracle never drift."""
        from controllers.threetank_dynamics import dynamics, build_params, NUMERIC_OPS
        if not hasattr(self, '_dynamics_params'):
            self._dynamics_params = build_params(self)
        self._last_t_res = x[7]  # cache for ideal_power
        self._dynamics_params["t_ambient"] = env.get("t_amb", self.t_ambient)
        return dynamics(x, act["pumps"][0], list(act["valves"]), act["heaters"][0],
                        self._dynamics_params, NUMERIC_OPS)

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
        # Minor/#6: reuse the ONE pump-curve implementation (compute_flows) — the
        # inline mirror was exactly the drift the consolidation was meant to end.
        # q_pump depends only on the duty + reservoir availability (nominal here).
        from controllers.threetank_dynamics import compute_flows, build_params, NUMERIC_OPS
        if not hasattr(self, '_dynamics_params'):
            self._dynamics_params = build_params(self)
        f = compute_flows([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0],
                          act["pumps"][0], [0.0, 0.0, 0.0], 0.0,
                          self._dynamics_params, NUMERIC_OPS)
        q_pump = float(f["q_pump"])
        rho_cp = self.rho * self.cp
        t_amb = env["t_amb"]
        # M2/#6: inflow temp from the MEASURED T3 (the reservoir return feeds the
        # pump). The old _last_t_res cache only updated inside derivatives(), so
        # scorer-only instances (validate_policy / rollout_report / manual_gui)
        # froze it at t_supply and ideal_w was ~93 kW vs a 2 kW heater — the
        # excess-energy KPI registered nothing on those paths. T3 slightly
        # overestimates T_res (the reservoir cools a little), so the floor errs
        # low and excess flags waste early — the conservative direction.
        t_inflow = temps[2] if len(temps) >= 3 else self._last_t_res
        return max(0.0, rho_cp * q_pump * (t_sp[0] - t_inflow) + self.ua * (t_sp[0] - t_amb))

    def clamp_state(self, x):
        return x
