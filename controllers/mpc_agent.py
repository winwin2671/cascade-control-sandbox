"""MPCAgent — copied verbatim from AIO-Gym's aiogym/baselines.py.

Successive-linearization, velocity-form (M=1) constrained MPC: at each solve it
finite-difference-linearizes the plant model about (x0, u0), builds a discrete
LTI perturbation model, and solves a box-constrained QP (cyclic coordinate
descent) over the move. Self-contained (numpy only, no CasADi). Point it at any
model implementing the interface (see threetank_model.py).
"""
from __future__ import annotations

import numpy as np


class MPCAgent:
    """Successive-linearization, velocity-form (M=1) constrained MPC — port of mpc.js."""
    name = "MPC"

    def __init__(self, model, Ts=0.5, P=40, move_supp=0.8, du_max=0.15,
                 cv_scale_level=0.1, cv_scale_temp=12.0):
        self.m = model
        self.nP, self.nV, self.nH = model.actuator_counts()
        self.nu = self.nP + self.nV + self.nH
        self.nx = len(model.initial_state())
        self.ctrl = model.controlled_levels()
        self.Ts, self.P, self.move_supp, self.du_max = Ts, P, move_supp, du_max
        cs = model.cv_scales() if hasattr(model, "cv_scales") else {}
        self.csl, self.cst = cs.get("level", cv_scale_level), cs.get("temp", cv_scale_temp)
        self.reset()

    def reset(self):
        init = self.m.mpc_init() if hasattr(self.m, "mpc_init") else None
        self.u = (np.asarray(init, dtype=np.float64) if init is not None
                  else np.array([0.35] * self.nP + [0.5] * self.nV + [0.0] * self.nH, dtype=np.float64))
        self._clock = 1e9

    def _unpack(self, u):
        return {"pumps": list(u[:self.nP]), "valves": list(u[self.nP:self.nP + self.nV]),
                "heaters": list(u[self.nP + self.nV:])}

    def _toX(self, meas):
        s = self.m.scenario
        if s == "cstr":
            return np.array([meas["conc"][0], meas["temps"][0]], dtype=np.float64)
        if s == "hvac":
            return np.array([meas["temps"][0], meas["temps"][1]], dtype=np.float64)
        if s == "heater":
            tfb = meas.get("tfb", [700.0])[0]
            return np.array([tfb, meas["temps"][0], meas["levels"][0]], dtype=np.float64)
        x = np.zeros(self.nx)
        for i in range(self.m.n):
            x[2 * i] = meas["levels"][i]
            x[2 * i + 1] = meas["temps"][i]
        return x

    def _cv(self, x):
        lv, tp = self.m.levels_temps(list(x))
        return np.array([lv[i] for i in self.ctrl] + list(tp), dtype=np.float64)

    def _wcv(self):
        return np.array([1 / self.csl ** 2] * len(self.ctrl) + [1 / self.cst ** 2] * self.m.n)

    def compute(self, meas, sp, dt):
        self._clock += dt
        if self._clock >= self.Ts:
            self._clock = 0.0
            self._solve(meas, sp)
        return self._unpack(self.u)

    def _solve(self, meas, sp):
        m, nx, nu, P, Ts = self.m, self.nx, self.nu, self.P, self.Ts
        env = {"t_cold": meas["t_cold"], "t_amb": meas["t_amb"], "extra_outflow": 0.0}
        x0 = self._toX(meas)
        u0 = self.u.copy()
        f = lambda x: np.asarray(m.derivatives(list(x), self._unpack(u0), env), dtype=np.float64)
        f0 = f(x0)
        eps = 1e-5
        Ad = np.eye(nx)
        Bd = np.zeros((nx, nu))
        for j in range(nx):
            xp = x0.copy(); xp[j] += eps
            xm = x0.copy(); xm[j] -= eps
            Ad[:, j] += (f(xp) - f(xm)) / (2 * eps) * Ts
        for j in range(nu):
            up = u0.copy(); up[j] += eps
            um = u0.copy(); um[j] -= eps
            fp = np.asarray(m.derivatives(list(x0), self._unpack(up), env), dtype=np.float64)
            fm = np.asarray(m.derivatives(list(x0), self._unpack(um), env), dtype=np.float64)
            Bd[:, j] = (fp - fm) / (2 * eps) * Ts
        cv0 = self._cv(x0)
        nCV = len(cv0)
        C = np.zeros((nCV, nx))
        for j in range(nx):
            xp = x0.copy(); xp[j] += eps
            C[:, j] = (self._cv(xp) - cv0) / eps
        target = np.array([sp["h_sp"][i] for i in self.ctrl] + list(sp["t_sp"][:m.n]), dtype=np.float64)
        Wcv = self._wcv()
        c0 = (x0 + f0 * Ts) - Ad @ x0 - Bd @ u0
        xf = x0.copy()
        S = np.zeros((nx, nu))
        H = np.zeros((nu, nu))
        g = np.zeros(nu)
        for _ in range(P):
            xf = Ad @ xf + Bd @ u0 + c0
            S = Ad @ S + Bd
            G = C @ S                      # nCV x nu
            e = C @ xf - target           # nCV
            WG = Wcv[:, None] * G
            H += G.T @ WG
            g += G.T @ (Wcv * e)
        H += self.move_supp * np.eye(nu)
        # box-QP via cyclic coordinate descent: solve each move within its own
        # rate+box bound so a saturated MV doesn't poison the others.
        lo = np.maximum(-self.du_max, 0.0 - u0)
        hi = np.minimum(self.du_max, 1.0 - u0)
        du = np.zeros(nu)
        for _ in range(6):
            moved = 0.0
            for j in range(nu):
                s_ = g[j] + sum(H[j, k] * du[k] for k in range(nu) if k != j)
                cand = float(np.clip(-s_ / (H[j, j] or 1e-9), lo[j], hi[j]))
                moved = max(moved, abs(cand - du[j]))
                du[j] = cand
            if moved < 1e-6:
                break
        self.u = np.clip(u0 + du, 0.0, 1.0)
