#!/usr/bin/env python3
"""aio_bridge_env.py — Gymnasium env that bridges an RL agent to the 3-tank
cascade plant via IA2 (and falls back to direct Modbus).

Architecture (matches the project goal flow):

    RL agent  --(Gym API)-->  CascadeBridgeEnv  --(HTTP /api/runtime/...)-->  IA2
        IA2  --(iomap)-->  Modbus TCP 127.0.0.1:5020  -->  mock_cabinet.py (plant)

This is the cascade-control-sandbox counterpart of AIO-Gym's
``aiogym/env.py`` (AIOGymNativeEnv): same Gymnasium contract — Box
observation (3 levels + 3 temperatures, engineering units) and Box action
(two pump fractions in [0, 1]) — but the plant is the *external* IA2 +
mock_cabinet instead of an in-process numpy model.

Two interchangeable backends (selected via ``backend=`` / ``--backend``):
  * ``ia2``    — reads observations from ``GET /api/runtime/snapshot`` and
                 writes actions with ``POST /api/runtime/variables/{name}``
                 (the "data endpoints" the plan calls for).  Requires IA2 to
                 be running with the ``ia2_project/`` project loaded and the
                 PROGRAM started (``cs run``).  This is the real bridge.
  * ``modbus`` — talks straight to mock_cabinet.py (pymodbus).  Used for
                 standalone testing without IA2 in the loop.
  * ``auto``   — use ``ia2`` when its /api/health answers, else ``modbus``.

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


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class Backend:
    """Read all registers (raw uint16, keyed by register name) and write one
    actuator register (by register name)."""

    def read_raw(self) -> dict[str, int]: ...
    def write_actuator(self, name: str, value: int) -> None: ...
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

    def write_actuator(self, name: str, value: int) -> None:
        idx = self.addr_names.index(name)
        r = self.client.write_registers(self.addr_base + idx, [int(value)],
                                        device_id=self.unit)
        if r.isError():
            raise RuntimeError(f"Modbus write error: {r}")

    def close(self) -> None:
        self.client.close()


class IA2Backend(Backend):
    """IA2 HTTP data-endpoint backend.

    Observations: GET /api/runtime/snapshot  -> VarSnapshot
        {timestamp_us, scan_count, vars: [{name, type_name, value(str)}]}
    Actions: POST /api/runtime/variables/{name}  body {"value": <i32>}
        (a between-scan variable write; the iomap forwards it to the cabinet).
    Variable-name format from IA2 is learned from the snapshot itself so we
    write back with whatever qualified name IA2 uses.
    """

    def __init__(self, server_url: str, project: str | None):
        self.base = server_url.rstrip("/")
        self.project = project
        health = _http_json("GET", f"{self.base}/api/health", timeout=1.5)
        if not health or health.get("status") not in ("ok", None):
            raise RuntimeError(f"IA2 health check failed: {health}")
        self._full_names: dict[str, str] = {}

    def _hdr(self) -> dict:
        return {"X-IA2-Project": self.project} if self.project else {}

    def _snapshot_vars(self) -> list[dict]:
        snap = _http_json("GET", f"{self.base}/api/runtime/snapshot",
                          headers=self._hdr(), timeout=2.0)
        if not snap or not snap.get("vars"):
            raise RuntimeError(
                "IA2 runtime snapshot is empty — open ia2_project/ and start "
                f"the program (e.g. `cs --server {self.base} run --program ThreeTank`)."
            )
        return snap["vars"]

    @staticmethod
    def _suffix(name: str) -> str:
        low = name.lower()
        return low.rsplit(".", 1)[-1] if "." in low else low

    def read_raw(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for v in self._snapshot_vars():
            full = v["name"]
            self._full_names[self._suffix(full)] = full
            try:
                out[self._suffix(full)] = int(str(v["value"]).strip())
            except ValueError:
                continue
        return out

    def write_actuator(self, name: str, value: int) -> None:
        full = self._full_names.get(name, name)
        url = (f"{self.base}/api/runtime/variables/"
               f"{urllib.parse.quote(full, safe='')}")
        _http_json("POST", url, body={"value": int(value)},
                   headers=self._hdr(), timeout=2.0)

    def close(self) -> None:
        pass


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
                 backend: str = "auto", control_dt: float = 0.5):
        super().__init__()
        self.config = load_config(config) if not isinstance(config, dict) else config
        self.control_dt = float(control_dt)

        regs = _registers_by_address(self.config)
        self.addr_base = int(regs[0]["address"])
        self.addr_names = [r["name"] for r in regs]                 # ordered by address
        self.sensor_names = list(self.config["sensors"])
        self.actuator_names = list(self.config["actuators"])
        self.scales = {r["name"]: float(r["scale"]) for r in regs}
        self.setpoints = {k: float(v) for k, v in
                          self.config["control"]["setpoints_m"].items()}
        self._actuator_max_raw = 10000  # ACT_MAX in mock_cabinet.py

        self.backend: Backend = self._make_backend(backend)

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
        be = ModbusBackend(m["host"], int(m["port"]), int(m["unit_id"]),
                           self.addr_base, self.addr_names)
        LOG.info("backend = Modbus (%s:%s)", m["host"], m["port"])
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

    def _action_to_raw(self, action) -> dict[str, int]:
        a = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        out = {}
        for i, name in enumerate(self.actuator_names):
            raw = int(round(a[i] / self.scales[name]))
            out[name] = max(0, min(raw, self._actuator_max_raw))
        return out

    def _reward(self, action, obs) -> tuple[float, dict]:
        sidx = {n: i for i, n in enumerate(self.sensor_names)}
        levels = {n: float(obs[sidx[n]]) for n in self.setpoints}  # level names
        track = sum((levels[n] - self.setpoints[n]) ** 2 for n in levels)
        energy = 0.01 * float(np.sum(np.asarray(action)))
        reward = float(-(track + energy))
        info = {
            "levels_m": levels,
            "track_mse": track,
            "action": np.asarray(action, dtype=np.float32).tolist(),
        }
        return reward, info

    # ---- Gym API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        for name in self.actuator_names:
            self.backend.write_actuator(name, 0)
        time.sleep(self.control_dt)
        obs = self._decode_obs(self.backend.read_raw())
        return obs, {"note": "actuators zeroed; plant state continues (no hardware reset)"}

    def step(self, action):
        for name, value in self._action_to_raw(action).items():
            self.backend.write_actuator(name, value)
        time.sleep(self.control_dt)
        obs = self._decode_obs(self.backend.read_raw())
        reward, info = self._reward(action, obs)
        return obs, reward, False, False, info

    def close(self):
        self.backend.close()


# --------------------------------------------------------------------------- #
# Demo: random-policy rollout to exercise the full loop end to end.
# --------------------------------------------------------------------------- #
def _demo(backend: str, steps: int, control_dt: float):
    env = CascadeBridgeEnv(backend=backend, control_dt=control_dt)
    obs, info = env.reset()
    LOG.info("reset obs = %s", np.round(obs, 3))
    rewards = []
    for k in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        if k % 4 == 0 or k == steps - 1:
            lv = info["levels_m"]
            LOG.info("step %3d  act=%s  levels(m) h1=%.3f h2=%.3f h3=%.3f  r=%.4f",
                     k, np.round(action, 2),
                     lv.get("tank1_level", float("nan")),
                     lv.get("tank2_level", float("nan")),
                     lv.get("tank3_level", float("nan")), reward)
    env.close()
    LOG.info("rollout done — mean reward = %.4f over %d steps", np.mean(rewards), steps)


def main():
    ap = argparse.ArgumentParser(description="AIO bridge Gym env (3-tank cascade).")
    ap.add_argument("--backend", choices=("auto", "ia2", "modbus"), default="auto")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--control-dt", type=float, default=0.5)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.verbose:
        logging.getLogger("pymodbus").setLevel(logging.WARNING)
    _demo(args.backend, args.steps, args.control_dt)


if __name__ == "__main__":
    main()
