"""NMPC oracle — CasADi + IPOPT nonlinear MPC, adapted from AIO-Gym's oracle.py.

Direct multiple-shooting transcription: RK4 over each control interval, IPOPT NLP.
The symbolic plant dynamics (`_f_threetank`) mirror mock_cabinet.py exactly (a
perfect-model oracle): Torricelli inter-tank flows + the first-law thermal energy
balance, with `ca.fmax`/`ca.sqrt` smoothing so IPOPT doesn't stall at the flow
kinks. Track mode only (setpoint tracking); the AIO-Gym economic mode + ECON table
were dropped (not relevant here).

Copied from aiogym/oracle.py (NMPCOracle + OracleAgent); `make_model` replaced
with our ThreeTankModel, and `_DYN` extended with `_f_threetank`.
"""
from __future__ import annotations

import numpy as np

try:
    import casadi as ca
    _HAVE_CASADI = True
except Exception:                       # pragma: no cover
    _HAVE_CASADI = False

from controllers.threetank_model import ThreeTankModel

RHO_CP = 1000.0 * 4186.0


# ----------------------------------------------------------------------------
# Symbolic continuous dynamics dx/dt = f(x, u, d, p) — mirror mock_cabinet.py.
# x = [h1, T1, h2, T2, h3, T3]; u = [p1, p2, heater1, heater2, heater3] in [0,1];
# d = [t_cold, t_amb]; p = ThreeTankModel.p.
# ----------------------------------------------------------------------------
def _flow_sym(h_from, h_to, coeff, p):
    """Smooth Torricelli flow + its forward/backward parts (well-mixed advection).

    q = coeff*S_PIPE*sqrt(2g)*sign(dh)*sqrt(|dh|), split into forward (dh>0) and
    backward (dh<0) via ca.fmax so the advection term carries the upstream temp.
    """
    dh = h_from - h_to
    k = coeff * p["S_PIPE"] * ca.sqrt(2 * p["G"])
    # floor the sqrt argument at 1e-9 (NOT 0): sqrt(0) has an infinite derivative
    # -> NaN in IPOPT's constraint Jacobian. (Same trick as AIO-Gym's _f_cascade.)
    fwd = k * ca.sqrt(ca.fmax(dh, 1e-9))
    bwd = k * ca.sqrt(ca.fmax(-dh, 1e-9))
    return fwd - bwd, fwd, bwd


def _dT_sym(T, h, q_heat, adv, t_amb, p):
    h = ca.fmax(h, p["h_floor"])
    m_cp = p["rho"] * p["A_TANK"] * h * p["cp"]
    q_loss = p["ua"] * (T - t_amb)
    return (q_heat - q_loss) / m_cp + adv / (p["A_TANK"] * h)


def _f_threetank(x, u, d, p):
    t_cold, t_amb = d[0], d[1]
    h1, T1, h2, T2, h3, T3 = x[0], x[1], x[2], x[3], x[4], x[5]
    q1 = u[0] * p["q_max"]
    q2 = u[1] * p["q_max"]
    q_12, f12, b12 = _flow_sym(h1, h2, p["A1"], p)
    q_32, f32, b32 = _flow_sym(h3, h2, p["A3"], p)
    q_drain, _, _ = _flow_sym(h2, 0.0, p["A2"], p)
    dh1 = (q1 - q_12) / p["A_TANK"]
    dh3 = (q2 - q_32) / p["A_TANK"]
    dh2 = (q_12 + q_32 - q_drain) / p["A_TANK"]
    Qh = [u[2 + i] * p["q_heat_max"] for i in range(3)]
    adv1 = q1 * (t_cold - T1) + b12 * (T2 - T1)
    adv2 = f12 * (T1 - T2) + f32 * (T3 - T2)
    adv3 = q2 * (t_cold - T3) + b32 * (T2 - T3)
    dT1 = _dT_sym(T1, h1, Qh[0], adv1, t_amb, p)
    dT2 = _dT_sym(T2, h2, Qh[1], adv2, t_amb, p)
    dT3 = _dT_sym(T3, h3, Qh[2], adv3, t_amb, p)
    return ca.vertcat(dh1, dT1, dh2, dT2, dh3, dT3)


_DYN = {"threetank": _f_threetank}


class NMPCOracle:
    """CasADi + IPOPT nonlinear MPC (multiple-shooting, RK4, tracking mode)."""

    def __init__(self, horizon=20, control_dt=0.5, du_max=0.4,
                 q_temp=1.0, q_level=50.0, r_move=0.05):
        if not _HAVE_CASADI:
            raise RuntimeError("casadi not installed — pip install casadi")
        self.scenario = "threetank"
        self.model = ThreeTankModel()
        self.p = self.model.p
        self.N = int(horizon)
        self.dt = float(control_dt)
        self.du_max = du_max
        nP, nV, nH = self.model.actuator_counts()
        self.nP, self.nV, self.nH = nP, nV, nH
        self.nu = nP + nV + nH
        self.nx = len(self.model.initial_state())
        self.q_temp, self.q_level, self.r_move = q_temp, q_level, r_move
        self.t_safe = 70.0          # match the L5 shield's high-temp cutoff
        self.u_prev = np.full(self.nu, 0.5)
        self._build()

    def _rk4(self, x, u, d):
        f = lambda xx: _DYN[self.scenario](xx, u, d, self.p)
        nsub = max(1, min(6, int(round(self.dt / self.model.dt_micro))))
        h = self.dt / nsub
        for _ in range(nsub):
            k1 = f(x); k2 = f(x + 0.5 * h * k1); k3 = f(x + 0.5 * h * k2); k4 = f(x + h * k3)
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return x

    def _temp_idx(self, i):
        return 2 * i + 1            # interleaved [h1,T1,h2,T2,h3,T3]

    def _stage_cost(self, x, u, sp, d):
        n = self.model.n
        c = 0
        for i in range(n):
            c += self.q_temp * (x[self._temp_idx(i)] - sp["t_sp"][i]) ** 2
        for j, i in enumerate(self.model.controlled_levels()):
            c += self.q_level * (x[2 * i] - sp["h_sp"][i]) ** 2
        return c

    def _build(self):
        N, nx, nu = self.N, self.nx, self.nu
        opti = ca.Opti()
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)
        x0 = opti.parameter(nx)
        d = opti.parameter(2)
        u_prev = opti.parameter(nu)
        tsp = opti.parameter(self.model.n)
        hsp = opti.parameter(self.model.n)
        sp = {"t_sp": [tsp[i] for i in range(self.model.n)],
              "h_sp": [hsp[i] for i in range(self.model.n)]}
        J = 0
        opti.subject_to(X[:, 0] == x0)
        slack = opti.variable(1, N)                                    # soft cap slack
        opti.subject_to(slack >= 0)
        for k in range(N):
            opti.subject_to(X[:, k + 1] == self._rk4(X[:, k], U[:, k], d))
            opti.subject_to(opti.bounded(0.0, U[:, k], 1.0))
            up = u_prev if k == 0 else U[:, k - 1]
            opti.subject_to(opti.bounded(-self.du_max, U[:, k] - up, self.du_max))
            for i in range(self.model.n):
                opti.subject_to(X[self._temp_idx(i), k + 1] <= self.t_safe + slack[0, k])
            J += self._stage_cost(X[:, k], U[:, k], sp, d) + self.r_move * ca.sumsqr(U[:, k] - up)
        J += 1e4 * ca.sumsqr(slack)
        opti.minimize(J)
        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 300,
                              "ipopt.acceptable_tol": 1e-4})
        self.opti, self.X, self.U = opti, X, U
        self.par = {"x0": x0, "d": d, "u_prev": u_prev, "tsp": tsp, "hsp": hsp}

    def reset(self):
        self.u_prev = np.full(self.nu, 0.5)

    def solve(self, x, t_cold, t_amb, t_sp, h_sp):
        o = self.opti
        o.set_value(self.par["x0"], np.asarray(x, float))
        o.set_value(self.par["d"], [t_cold, t_amb])
        o.set_value(self.par["u_prev"], self.u_prev)
        o.set_value(self.par["tsp"], np.asarray(t_sp, float))
        o.set_value(self.par["hsp"], np.asarray(
            [h_sp[i] if i < len(h_sp) else 0.0 for i in range(self.model.n)], float))
        try:
            o.set_initial(self.U, np.tile(self.u_prev.reshape(-1, 1), (1, self.N)))
            o.set_initial(self.X, np.tile(np.asarray(x, float).reshape(-1, 1), (1, self.N + 1)))
            sol = o.solve()
            u = np.clip(sol.value(self.U)[:, 0], 0.0, 1.0)
        except Exception:
            u = self.u_prev                                # keep last on solver failure
        self.u_prev = np.asarray(u, float).reshape(-1)
        return {"pumps": list(self.u_prev[:self.nP]),
                "valves": list(self.u_prev[self.nP:self.nP + self.nV]),
                "heaters": list(self.u_prev[self.nP + self.nV:])}


class OracleAgent:
    """Adapts NMPCOracle to the agent interface compute(meas, sp, dt)."""
    name = "NMPC-oracle"

    def __init__(self, **kw):
        self.orc = NMPCOracle(**kw)
        self.model = self.orc.model

    def reset(self):
        self.orc.reset()

    def _x_from_meas(self, meas):
        x = []                                             # interleave h, T
        for i in range(self.model.n):
            x += [meas["levels"][i], meas["temps"][i]]
        return x

    def compute(self, meas, sp, dt):
        return self.orc.solve(self._x_from_meas(meas), meas["t_cold"], meas["t_amb"],
                              sp["t_sp"], sp["h_sp"])
