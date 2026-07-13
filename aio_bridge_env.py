#!/usr/bin/env python3
"""aio_bridge_env.py — Gymnasium env that bridges an RL agent to the 3-tank
cascade plant via IA2 (and falls back to direct Modbus).

Architecture (matches the project goal flow):

    RL agent  --(Gym API)-->  CascadeBridgeEnv  --(HTTP /api/...)-->  IA2
        IA2  --(iomap)-->  Modbus TCP 127.0.0.1:5020  -->  mock_cabinet.py (plant)

This is the cascade-control-sandbox counterpart of AIO-Gym's
``aiogym/env.py`` (AIOGymNativeEnv): same Gymnasium contract — Box
observation (3 levels + 3 temperatures, engineering units) and Box action
(pump fractions in [0, 1]) — but the plant is the *external* IA2 +
mock_cabinet instead of an in-process numpy model.

Backends (selected via ``backend=`` / ``--backend``):
  * ``ia2``          — dev server: GET /api/runtime/snapshot (obs) +
                       POST /api/runtime/variables/{name} (actions).
  * ``edge[:name]``  — edge runtime (G4): GET /api/edges/{name}/status
                       (obs via .last_snapshot.vars) + POST
                       /api/edges/{name}/runtime/write body {name,value}
                       (the edge's body-addressed write, proxied by the dev
                       server over SSH).  Needs a registered, deployed edge.
  * ``modbus``       — talks straight to mock_cabinet.py (no IA2 in the loop).
  * ``auto``         — ``ia2`` when /api/health answers, else ``modbus``.

All register names, addresses, scales, and setpoints come from
ia2_config.json — the single contract shared with mock_cabinet.py and the
IA2 iomap — so this env stays in lockstep with them.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

LOG = logging.getLogger("aio_bridge")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else Path(__file__).resolve().parent / "ia2_config.json"
    with open(p) as fh:
        return json.load(fh)


def _registers_by_address(config: dict) -> list[dict]:
    return sorted(config["registers"], key=lambda r: r["address"])


# --------------------------------------------------------------------------- #
# HTTP helper (stdlib only — no extra dependency)
# --------------------------------------------------------------------------- #
def _http_json(method: str, url: str, body=None, headers: dict | None = None,
               timeout: float = 2.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:400]
        raise RuntimeError(f"{method} {url} -> HTTP {e.code}: {detail}") from None


def _suffix(name: str) -> str:
    """Lowercased unqualified variable name (handles POU/instance qualifiers)."""
    low = name.lower()
    return low.rsplit(".", 1)[-1] if "." in low else low


def _parse_vars(vars_list: list[dict], full_names: dict[str, str]) -> dict[str, int]:
    """Shared ``VarSnapshot.vars`` parser used by the IA2 and edge backends.

    Returns ``{suffix: int value}`` and fills ``full_names`` (suffix -> the
    exact name the runtime reported) so writes address what IA2 expects.
    Values are strings in the snapshot ("3370") -> int here.
    """
    out: dict[str, int] = {}
    for v in vars_list:
        full = v["name"]
        suf = _suffix(full)
        full_names[suf] = full
        try:
            out[suf] = int(str(v["value"]).strip())
        except (ValueError, TypeError):
            continue
    return out


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class Backend:
    """Read all registers (raw uint16, keyed by register name) and write any
    register/variable (actuators, reset_cmd, init_h*) by name."""

    writes_via_plc = False  # True for IA2/edge: writes reach actuators through the PLC

    def read_raw(self) -> dict[str, int]: ...
    def write_register(self, name: str, value: int) -> None: ...
    def close(self) -> None: ...


class ModbusBackend(Backend):
    """Direct pymodbus client to mock_cabinet.py (no IA2 in the loop)."""

    def __init__(self, host: str, port: int, unit_id: int,
                 addr_base: int, addr_names: list[str]):
        from pymodbus.client import ModbusTcpClient  # local import
        self.unit = unit_id
        self.addr_base = addr_base
        self.addr_names = addr_names
        self.client = ModbusTcpClient(host=host, port=port)
        if not self.client.connect():
            raise RuntimeError(f"cannot connect to cabinet at {host}:{port}")

    def read_raw(self) -> dict[str, int]:
        r = self.client.read_holding_registers(
            self.addr_base, count=len(self.addr_names), device_id=self.unit)
        if r.isError():
            raise RuntimeError(f"Modbus read error: {r}")
        return {n: int(v) for n, v in zip(self.addr_names, r.registers)}

    def write_register(self, name: str, value: int) -> None:
        idx = self.addr_names.index(name)
        r = self.client.write_registers(self.addr_base + idx, [int(value)],
                                        device_id=self.unit)
        if r.isError():
            raise RuntimeError(f"Modbus write error: {r}")

    def close(self) -> None:
        self.client.close()


class _IA2HttpBase(Backend):
    """Shared HTTP plumbing for the IA2 dev-server and edge backends."""

    writes_via_plc = True  # writes go to the PLC *_req vars -> through the L5 shield

    def __init__(self, server_url: str, project: str | None):
        self.base = server_url.rstrip("/")
        self.project = project
        health = _http_json("GET", f"{self.base}/api/health", timeout=1.5)
        if not health or health.get("status") not in ("ok", None):
            raise RuntimeError(f"IA2 health check failed: {health}")
        self._full_names: dict[str, str] = {}

    def _hdr(self) -> dict:
        return {"X-IA2-Project": self.project} if self.project else {}

    def write_register(self, name: str, value: int) -> None:
        full = self._full_names.get(name, name)
        self._post_write(full, int(value))

    def close(self) -> None:
        pass

    # subclasses supply the variable-source + write-route specifics:
    def _vars(self) -> list[dict]: ...
    def _post_write(self, full_name: str, value: int) -> None: ...

    def read_raw(self) -> dict[str, int]:
        return _parse_vars(self._vars(), self._full_names)


class IA2Backend(_IA2HttpBase):
    """Dev-server backend.

    Observations: GET /api/runtime/snapshot  -> VarSnapshot
        {timestamp_us, scan_count, vars: [{name, type_name, value(str)}]}
    Actions: POST /api/runtime/variables/{name}  body {"value": <i32>}
        (a between-scan variable write; the iomap forwards it to the cabinet).
    """

    def _vars(self) -> list[dict]:
        snap = _http_json("GET", f"{self.base}/api/runtime/snapshot",
                          headers=self._hdr(), timeout=2.0)
        if not snap or not snap.get("vars"):
            raise RuntimeError(
                "IA2 runtime snapshot is empty — open ia2_project/ and start "
                f"the program (e.g. `cs --server {self.base} run --program ThreeTank`)."
            )
        return snap["vars"]

    def _post_write(self, full_name: str, value: int) -> None:
        url = (f"{self.base}/api/runtime/variables/"
               f"{urllib.parse.quote(full_name, safe='')}")
        _http_json("POST", url, body={"value": value},
                   headers=self._hdr(), timeout=2.0)


class EdgeBackend(_IA2HttpBase):
    """Edge-runtime backend via the dev server's SSH proxy (addresses G4).

    For deployments where the project runs on a remote edge (``ia2-runtime``)
    rather than the local dev server. The edge runtime exposes a
    body-addressed write route (vs the dev server's path-addressed one); the
    dev server proxies it over SSH.

    Observations: GET /api/edges/{name}/status -> .last_snapshot.vars
                  (same VarSnapshot.vars shape as the dev server's snapshot).
    Actions:      POST /api/edges/{name}/runtime/write body {"name", "value"}
                  (the edge's body-addressed write).

    Cannot be exercised without a registered, deployed edge (``cs edge`` +
    ``cs deploy``); the route shapes above are verified against the IA2
    source (crates/server/src/edges.rs, crates/runtime/src/main.rs).
    """

    def __init__(self, server_url: str, project: str | None, edge_name: str):
        super().__init__(server_url, project)
        self.edge = edge_name

    def _vars(self) -> list[dict]:
        status = _http_json("GET", f"{self.base}/api/edges/{self.edge}/status",
                            headers=self._hdr(), timeout=4.0)
        snap = (status or {}).get("last_snapshot")
        if not snap or not snap.get("vars"):
            raise RuntimeError(
                f"edge '{self.edge}' /status has no last_snapshot — is the project "
                f"deployed and running on the edge? (cs edge create / cs deploy)"
            )
        return snap["vars"]

    def _post_write(self, full_name: str, value: int) -> None:
        _http_json("POST", f"{self.base}/api/edges/{self.edge}/runtime/write",
                   body={"name": full_name, "value": value},
                   headers=self._hdr(), timeout=4.0)


# --------------------------------------------------------------------------- #
# Gymnasium environment
# --------------------------------------------------------------------------- #
class CascadeBridgeEnv(gym.Env):
    """3-tank cascade control env over IA2 (or direct Modbus).

    observation = sensor registers in engineering units (default order
                  tank1_level, tank1_temp, tank2_level, tank2_temp,
                  tank3_level, tank3_temp) — driven by ia2_config.json.
    action      = actuator fractions in [0, 1] (actuator1, actuator2).
    reward      = -(level tracking error vs setpoints + small action cost).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict | str | Path | None = None,
                 backend: str = "auto", control_dt: float = 0.5, mode: str = "rl",
                 port: int | None = None):
        super().__init__()
        self.config = load_config(config) if not isinstance(config, dict) else config
        self.control_dt = float(control_dt)
        self._port_override = port

        regs = _registers_by_address(self.config)
        self.addr_base = int(regs[0]["address"])
        self.addr_names = [r["name"] for r in regs]                 # ordered by address
        self.sensor_names = list(self.config["sensors"])
        self.actuator_names = list(self.config["actuators"])
        self.scales = {r["name"]: float(r["scale"]) for r in regs}
        self.setpoints = {k: float(v) for k, v in
                          self.config["control"]["setpoints_m"].items()}
        self.temp_setpoints = {k: float(v) for k, v in
                               self.config["control"].get("setpoints_c", {}).items()}
        rw = self.config["control"].get("reward_weights", {})
        self.reward_weights = {
            "level": float(rw.get("level", 1.0)),
            "temp": float(rw.get("temp", 0.002)),
            "action": float(rw.get("action", 0.01)),
        }
        self._reset_nonce = 0

        self.backend: Backend = self._make_backend(backend)
        self.mode = mode.lower()
        self._mode_int = {"manual": 0, "pid": 1, "mpc": 2, "rl": 3}.get(self.mode, 3)
        # PLC-mode write targets (what the agent writes each step). Modbus backend
        # has no PLC -> drives the cabinet registers directly (mode ignored).
        self._write_names, self._write_max = self._write_targets()
        if self.backend.writes_via_plc:
            self.backend.write_register("mode", self._mode_int)  # PLC CASE selector
            LOG.info("mode = %s (%d)", self.mode, self._mode_int)

        sreg = {r["name"]: r for r in regs}
        obs_lo = np.array([sreg[n]["min"] for n in self.sensor_names], dtype=np.float32)
        obs_hi = np.array([sreg[n]["max"] for n in self.sensor_names], dtype=np.float32)
        self.observation_space = spaces.Box(obs_lo, obs_hi, dtype=np.float32)
        self.action_space = spaces.Box(
            np.zeros(len(self.actuator_names), np.float32),
            np.ones(len(self.actuator_names), np.float32), dtype=np.float32,
        )

    # ---- backend selection ----
    def _make_backend(self, kind: str) -> Backend:
        # "edge" or "edge:<name>" -> edge-runtime backend via dev-server proxy.
        if kind.startswith("edge"):
            ia2 = self.config["ia2"]
            edge_name = kind.split(":", 1)[1] if ":" in kind else ia2.get("edge_name")
            if not edge_name:
                raise RuntimeError(
                    "--backend edge requires a name: use 'edge:<name>' or set "
                    "ia2.edge_name in ia2_config.json"
                )
            be = EdgeBackend(ia2["server_url"], ia2.get("project_name"), edge_name)
            LOG.info("backend = Edge (%s, edge=%s)", ia2["server_url"], edge_name)
            return be
        if kind in ("ia2", "auto"):
            try:
                ia2 = self.config["ia2"]
                be = IA2Backend(ia2["server_url"], ia2.get("project_name"))
                LOG.info("backend = IA2 (%s)", ia2["server_url"])
                return be
            except Exception as e:
                if kind == "ia2":
                    raise
                LOG.warning("IA2 backend unavailable (%s); using Modbus", e)
        m = self.config["modbus"]
        port = int(self._port_override) if self._port_override else int(m["port"])
        be = ModbusBackend(m["host"], port, int(m["unit_id"]),
                           self.addr_base, self.addr_names)
        LOG.info("backend = Modbus (%s:%s)", m["host"], port)
        return be

    # ---- conversions ----
    def _decode_obs(self, raw: dict[str, int]) -> np.ndarray:
        vals = []
        for name in self.sensor_names:
            if name not in raw:
                raise RuntimeError(
                    f"sensor '{name}' missing from backend read; "
                    f"got keys={list(raw)}"
                )
            vals.append(raw[name] * self.scales[name])
        return np.asarray(vals, dtype=np.float32)

    def _write_targets(self) -> tuple[list[str], list[int]]:
        """Var names + raw maxima the agent writes each step, by mode/backend."""
        if not self.backend.writes_via_plc:                      # modbus -> direct
            return list(self.actuator_names), [10000] * len(self.actuator_names)
        if self.mode == "manual":
            return (["manual_p1", "manual_p2", "manual_h1", "manual_h2", "manual_h3"],
                    [10000] * 5)
        if self.mode == "pid":
            return (["tank1_level_sp", "tank3_level_sp",
                     "tank1_temp_sp", "tank2_temp_sp", "tank3_temp_sp"],
                    [6000, 6000, 10000, 10000, 10000])
        return ([f"{n}_req" for n in self.actuator_names],       # mpc / rl
                [10000] * len(self.actuator_names))

    def _action_to_writes(self, action) -> dict[str, int]:
        a = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        return {name: max(0, min(int(round(a[i] * mx)), mx))
                for i, (name, mx) in enumerate(zip(self._write_names, self._write_max))}

    def setpoint_action(self) -> np.ndarray:
        """Config setpoints as a normalized [0,1] action (for the PID-mode demo)."""
        sp = self.config["control"]
        h_max = float(self.config["process"]["h_max_m"])
        return np.array([
            sp["setpoints_m"]["tank1_level"] / h_max,
            sp["setpoints_m"]["tank3_level"] / h_max,
            sp["setpoints_c"]["tank1_temp"] / 100.0,
            sp["setpoints_c"]["tank2_temp"] / 100.0,
            sp["setpoints_c"]["tank3_temp"] / 100.0,
        ], dtype=np.float32)

    def _reward(self, action, obs) -> tuple[float, dict]:
        sidx = {n: i for i, n in enumerate(self.sensor_names)}
        levels = {n: float(obs[sidx[n]]) for n in self.setpoints}            # level sensors
        temps = {n: float(obs[sidx[n]]) for n in self.temp_setpoints}        # temp sensors
        w = self.reward_weights
        track_l = sum((levels[n] - self.setpoints[n]) ** 2 for n in levels)
        track_t = sum((temps[n] - self.temp_setpoints[n]) ** 2 for n in temps)
        action_cost = w["action"] * float(np.sum(np.asarray(action)))
        reward = float(-(w["level"] * track_l + w["temp"] * track_t + action_cost))
        info = {
            "levels_m": levels,
            "temps_c": temps,
            "track_level_mse": track_l,
            "track_temp_mse": track_t,
            "action": np.asarray(action, dtype=np.float32).tolist(),
        }
        return reward, info

    # ---- Gym API ----
    def reset(self, *, seed=None, options=None):
        """Reset the plant to a sampled initial state (RL init-state distribution).

        Writes sampled initial levels to the init_h* registers, then pulses
        reset_cmd (rising edge). The cabinet snaps to init_h* and HOLDS while
        reset_cmd is asserted, so the obs read here are exactly the init levels.
        Releasing reset_cmd (-> 0) lets the cabinet resume stepping for step().
        """
        super().reset(seed=seed)
        for name in self._write_names:                   # neutral the mode's write vars
            self.backend.write_register(name, 0)
        info: dict = {}
        rcfg = self.config.get("reset")
        if rcfg:
            lo, hi = rcfg.get("init_level_range_m", [0.10, 0.50])
            init_levels: dict[str, float] = {}
            for name in rcfg.get("init_levels", []):
                level = float(self.np_random.uniform(lo, hi))
                mx = round(1.0 / self.scales[name])
                init_levels[name] = level
                self.backend.write_register(
                    name, max(0, min(int(round(level / self.scales[name])), mx)))
            info["init_levels_m"] = init_levels
            cmd = rcfg.get("command_register")
            if cmd:
                self._reset_nonce = (self._reset_nonce + 1) % 65536
                self.backend.write_register(cmd, self._reset_nonce)  # fresh nonce -> snap + hold
                time.sleep(self.control_dt)
                obs = self._decode_obs(self.backend.read_raw())
                self.backend.write_register(cmd, 0)                   # release -> resume
                return obs, info
        time.sleep(self.control_dt)
        return self._decode_obs(self.backend.read_raw()), info

    def step(self, action):
        for name, value in self._action_to_writes(action).items():
            self.backend.write_register(name, value)
        time.sleep(self.control_dt)
        obs = self._decode_obs(self.backend.read_raw())
        reward, info = self._reward(action, obs)
        return obs, reward, False, False, info

    def close(self):
        self.backend.close()


# --------------------------------------------------------------------------- #
# Demo: random-policy rollout to exercise the full loop end to end.
# --------------------------------------------------------------------------- #
def _demo(backend: str, steps: int, control_dt: float, mode: str):
    env = CascadeBridgeEnv(backend=backend, control_dt=control_dt, mode=mode)
    obs, info = env.reset()
    LOG.info("reset obs = %s  mode=%s", np.round(obs, 3), mode)
    rewards = []
    pid_act = env.setpoint_action() if mode == "pid" else None
    for k in range(steps):
        action = pid_act if pid_act is not None else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if k % 4 == 0 or k == steps - 1:
            lv = info["levels_m"]
            tp = info.get("temps_c", {})
            LOG.info("step %3d  act=%s  levels(m)=%.3f/%.3f/%.3f  "
                     "temps(C)=%.1f/%.1f/%.1f  r=%.4f",
                     k, np.round(action, 2),
                     lv.get("tank1_level", float("nan")),
                     lv.get("tank2_level", float("nan")),
                     lv.get("tank3_level", float("nan")),
                     tp.get("tank1_temp", float("nan")),
                     tp.get("tank2_temp", float("nan")),
                     tp.get("tank3_temp", float("nan")), reward)
    env.close()
    LOG.info("rollout done — mean reward = %.4f over %d steps", np.mean(rewards), steps)


def main():
    ap = argparse.ArgumentParser(description="AIO bridge Gym env (3-tank cascade).")
    ap.add_argument("--backend", default="auto",
                    help="auto | ia2 | modbus | edge | edge:<name> (default: auto)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--control-dt", type=float, default=0.5)
    ap.add_argument("--mode", default="rl",
                    help="control mode: manual | pid | mpc | rl (default rl; IA2 backend only)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        logging.getLogger("pymodbus").setLevel(logging.WARNING)
    _demo(args.backend, args.steps, args.control_dt, args.mode)


if __name__ == "__main__":
    main()
