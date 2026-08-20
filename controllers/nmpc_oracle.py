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

import logging

import numpy as np

try:
    import casadi as ca
    _HAVE_CASADI = True
except Exception:                       # pragma: no cover
    _HAVE_CASADI = False

from controllers.threetank_model import ThreeTankModel

LOG = logging.getLogger("nmpc_oracle")

RHO_CP = 1000.0 * 4186.0


# ----------------------------------------------------------------------------
# Symbolic continuous dynamics dx/dt = f(x, u, d, p) — mirror mock_cabinet.py.
# x = [h1, T1, h2, T2, h3, T3]; u = [pump, V-12, V-23, V-33, heater] in [0,1];
# d = [t_cold, t_amb]; p = ThreeTankModel.p.
# ----------------------------------------------------------------------------
# Minor/#6: the pre-consolidation _valve_flow_sym/_dT_sym mirrors were removed —
# _f_threetank delegates to the shared threetank_dynamics module.

def _f_threetank(x, u, d, p):
    """CasADi dynamics — delegates to the shared threetank_dynamics module."""
    from controllers.threetank_dynamics import dynamics, casadi_ops
    t_amb = d[1]
    p["t_ambient"] = t_amb
    dx = dynamics(x, u[0], [u[1], u[2], u[3]], u[4], p, casadi_ops(ca))
    return ca.vertcat(*dx)


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
        self.solve_fails = 0
        from controllers.threetank_dynamics import build_params
        self._dyn_params = build_params(self.model)
        self._build()

    def _rk4(self, x, u, d):
        f = lambda xx: _DYN[self.scenario](xx, u, d, self._dyn_params)
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
        # B2/#6: physical envelope on X — without it the solver iterates into
        # h = −gravity_drop (the sqrt-floor kink) and negative levels, which is
        # where the max-iter stalls came from. Levels [0, h_max]; temps sane.
        for i in range(self.model.n):
            opti.subject_to(opti.bounded(0.0, X[2 * i, :], self.model.h_max))
            opti.subject_to(opti.bounded(5.0, X[2 * i + 1, :], 90.0))
        opti.subject_to(opti.bounded(0.0, X[6, :], 0.5))               # h_res
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
        self.solve_fails = 0                               # B2/#6: surfaced, not silent

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
        except Exception as e:
            # B2/#6: a dead oracle must not masquerade as a working one — log the
            # failure (first few verbosely, then a running count) instead of the
            # old bare except that returned u_prev forever with no signal.
            self.solve_fails += 1
            if self.solve_fails <= 3 or self.solve_fails % 50 == 0:
                LOG.warning("IPOPT solve failed (%d total): %s — holding u_prev",
                            self.solve_fails, str(e).strip().splitlines()[-1][:120])
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
        # finite reservoir (unmeasured — nominal estimate)
        x += [0.30, meas.get("t_cold", 25.0)]  # h_res, T_res
        return x

    def compute(self, meas, sp, dt):
        return self.orc.solve(self._x_from_meas(meas), meas["t_cold"], meas["t_amb"],
                              sp["t_sp"], sp["h_sp"])
