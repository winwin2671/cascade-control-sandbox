#!/usr/bin/env python3
"""aio_bridge_env.py — Gymnasium env bridging an RL agent to the heated
serial-cascade + recirculation plant via IA2 (or direct Modbus).

Architecture:

    RL agent  --(Gym API)-->  CascadeBridgeEnv  --(HTTP /api/...)-->  IA2
        IA2  --(iomap)-->  Modbus TCP gateway 127.0.0.1:5020  -->  mock_cabinet.py (plant)

Same Gymnasium contract as AIO-Gym's AIOGymNativeEnv, but the plant is the
external IA2 + mock_cabinet. The cabinet is now MULTI-SLAVE behind an RTU-to-TCP
gateway with 32-bit float analog channels and FC02 discretes, so the Modbus
backend dispatches per register: (slave_id, function code, data type) come from
ia2_config.json — the single contract shared with mock_cabinet.py and the iomap.

Backends (selected via ``backend=`` / ``--backend``):
  * ``ia2``          — dev server snapshot (obs) + variable write (actions).
  * ``edge[:name]``  — edge runtime via the dev server's SSH proxy.
  * ``modbus``       — talks straight to mock_cabinet.py (no IA2 in the loop).
  * ``auto``         — ``ia2`` when /api/health answers, else ``modbus``.
"""
from __future__ import annotations

import argparse
import json
import logging
import struct
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
# 32-bit float register codec (Modbus "ABCD": big-endian bytes, high word first).
# MUST match mock_cabinet.pack_float_be — the mock decodes writes, the bridge
# encodes them (and decodes reads). Honors byte_order/word_order so a non-ABCD
# field module can be supported by swapping word order here only.
# --------------------------------------------------------------------------- #
def decode_float(regs, byte_order: str = "big", word_order: str = "big") -> float:
    """Decode two 16-bit registers to a float (default ABCD big-endian)."""
    hi, lo = int(regs[0]), int(regs[1])
    if word_order == "little":
        hi, lo = lo, hi
    raw = bytes(((hi >> 8) & 0xFF, hi & 0xFF, (lo >> 8) & 0xFF, lo & 0xFF))
    if byte_order == "little":
        raw = raw[::-1]
    return struct.unpack(">f", raw)[0]


def encode_float(v: float, byte_order: str = "big", word_order: str = "big") -> list[int]:
    """Encode a float to two 16-bit registers (default ABCD big-endian)."""
    b = struct.pack(">f", float(v))
    hi, lo = int.from_bytes(b[0:2], "big"), int.from_bytes(b[2:4], "big")
    if word_order == "little":
        hi, lo = lo, hi
    return [hi, lo]


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


def _parse_vars(vars_list: list[dict], full_names: dict[str, str]) -> dict:
    """Shared ``VarSnapshot.vars`` parser for the IA2 / edge backends.

    Returns ``{suffix: value}`` (int or float) and fills ``full_names`` so writes
    address the exact name the runtime reported. The new plant exposes float
    (REAL) POU variables, so values are parsed as int first, then float.
    """
    out: dict = {}
    for v in vars_list:
        full = v["name"]
        suf = _suffix(full)
        full_names[suf] = full
        s = str(v["value"]).strip()
        tn = str(v.get("type_name", "")).upper()
        if tn == "BOOL" or s in ("TRUE", "FALSE", "True", "False"):
            out[suf] = (s.upper() == "TRUE")
            continue
        try:
            out[suf] = int(s)
        except (ValueError, TypeError):
            try:
                out[suf] = float(s)
            except (ValueError, TypeError):
                continue
    return out


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class Backend:
    """Read all read-registers (keyed by name, engineering units) and write any
    register/variable (actuators, reset_cmd, init_h*) by name."""

    writes_via_plc = False  # True for IA2/edge: writes reach actuators through the PLC

    def read_raw(self) -> dict: ...
    def write_register(self, name: str, value) -> None: ...
    def close(self) -> None: ...


class ModbusBackend(Backend):
    """Direct pymodbus client to mock_cabinet.py (no IA2 in the loop).

    Multi-slave: every register carries its own (slave_id, fc, type, count,
    byte_order, word_order), so reads dispatch FC04 (input) / FC02 (discrete) /
    FC03 (holding) to the right slave and decode floats/bools/uint16 per type.
    Writes encode likewise (float -> 2 regs FC16, uint16 -> FC06, bool -> FC05).
    """

    def __init__(self, host: str, port: int, regs: list[dict]):
        from pymodbus.client import ModbusTcpClient  # local import
        self.regs_by_name = {r["name"]: r for r in regs}
        self.client = ModbusTcpClient(host=host, port=port)
        if not self.client.connect():
            raise RuntimeError(f"cannot connect to cabinet at {host}:{port}")

    def read_raw(self) -> dict:
        out: dict = {}
        # batch reads by (slave_id, fc) — each group is one contiguous block.
        groups: dict[tuple, list[dict]] = {}
        for r in self.regs_by_name.values():
            if r["direction"] != "read":
                continue
            groups.setdefault((r["slave_id"], r["fc"]), []).append(r)
        for (sid, fc), regs in groups.items():
            regs = sorted(regs, key=lambda r: r["address"])
            base = regs[0]["address"]
            count = sum(int(r["count"]) for r in regs)
            if fc == 4:
                rr = self.client.read_input_registers(base, count=count, device_id=sid)
                raw = rr.registers
            elif fc == 2:
                rr = self.client.read_discrete_inputs(base, count=count, device_id=sid)
                raw = rr.bits
            else:  # fc == 3
                rr = self.client.read_holding_registers(base, count=count, device_id=sid)
                raw = rr.registers
            if rr.isError():
                raise RuntimeError(f"Modbus read error (slave {sid} fc {fc}): {rr}")
            for r in regs:
                off = r["address"] - base
                cnt = int(r["count"])
                if r["type"] == "bool":
                    out[r["name"]] = bool(raw[off])
                elif r["type"] == "float":
                    out[r["name"]] = decode_float(raw[off:off + cnt],
                                                  r.get("byte_order", "big"),
                                                  r.get("word_order", "big"))
                else:  # uint16
                    out[r["name"]] = int(raw[off])
        return out

    def write_register(self, name: str, value) -> None:
        r = self.regs_by_name[name]
        sid, addr = r["slave_id"], r["address"]
        if r["type"] == "float":
            regs = encode_float(value, r.get("byte_order", "big"), r.get("word_order", "big"))
            rr = self.client.write_registers(addr, regs, device_id=sid)         # FC16
        elif r["type"] == "bool":
            rr = self.client.write_coil(addr, bool(value), device_id=sid)       # FC05
        else:  # uint16
            rr = self.client.write_register(addr, int(value), device_id=sid)    # FC06
        if rr.isError():
            raise RuntimeError(f"Modbus write error ({name}): {rr}")

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

    def write_register(self, name: str, value) -> None:
        full = self._full_names.get(name, name)
        self._post_write(full, self._i32_value(value))

    @staticmethod
    def _i32_value(value) -> int:
        """IA2 variable-write takes i32. REAL vars take their IEEE-754 bits (the VM
        slot is f32::to_bits); integer vars (UINT/INT/DINT/BOOL) take the value
        directly. Floats -> bit-cast, ints/bools -> as-is."""
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, float):
            return struct.unpack("<i", struct.pack("<f", float(value)))[0]
        return int(value)

    def close(self) -> None:
        pass

    # subclasses supply the variable-source + write-route specifics:
    def _vars(self) -> list[dict]: ...
    def _post_write(self, full_name: str, value) -> None: ...

    def read_raw(self) -> dict:
        return _parse_vars(self._vars(), self._full_names)


class IA2Backend(_IA2HttpBase):
    """Dev-server backend.

    Observations: GET /api/runtime/snapshot  -> VarSnapshot
        {timestamp_us, scan_count, vars: [{name, type_name, value(str)}]}
    Actions: POST /api/runtime/variables/{name}  body {"value": <number>}
        (a between-scan variable write; the iomap forwards it to the cabinet).
    """

    def __init__(self, server_url: str, project: str | None,
                 probe_runtime: bool = False):
        super().__init__(server_url, project)
        self._prev_scan_count = None   # C7: frozen-obs detection
        if probe_runtime:
            self.read_raw()  # surfaces a 409 / empty-snapshot inside _make_backend's try/except

    def _vars(self) -> list[dict]:
        snap = _http_json("GET", f"{self.base}/api/runtime/snapshot",
                          headers=self._hdr(), timeout=2.0)
        if not snap or not snap.get("vars"):
            raise RuntimeError(
                "IA2 runtime snapshot is empty — no program is loaded/running. "
                f"Start one with `cs run --server {self.base} --program ThreeTank` "
                "(or just `cs run --program ThreeTank` against the default server)."
            )
        sc = snap.get("scan_count")
        if sc is not None:
            sc = int(sc)
            if self._prev_scan_count is not None and sc == self._prev_scan_count:
                raise RuntimeError(
                    f"IA2 scan_count frozen at {sc} between steps — the runtime "
                    "is paused or the cabinet is dead. Stale observations would "
                    "poison training. Check: is mock_cabinet running? Is cs paused?")
            self._prev_scan_count = sc
        return snap["vars"]

    def _post_write(self, full_name: str, value) -> None:
        url = (f"{self.base}/api/runtime/variables/"
               f"{urllib.parse.quote(full_name, safe='')}")
        _http_json("POST", url, body={"value": value}, headers=self._hdr(), timeout=2.0)


class EdgeBackend(_IA2HttpBase):
    """Edge-runtime backend via the dev server's SSH proxy (addresses G4).

    Observations: GET /api/edges/{name}/status -> .last_snapshot.vars
    Actions:      POST /api/edges/{name}/runtime/write body {"name", "value"}
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
                "IA2 runtime snapshot is empty — no program is loaded/running. "
                f"Start one with `cs run --server {self.base} --program ThreeTank`.")
        return snap["vars"]

    def _post_write(self, full_name: str, value) -> None:
        _http_json("POST", f"{self.base}/api/edges/{self.edge}/runtime/write",
                   body={"name": full_name, "value": value},
                   headers=self._hdr(), timeout=4.0)


# --------------------------------------------------------------------------- #
# Gymnasium environment
# --------------------------------------------------------------------------- #
class CascadeBridgeEnv(gym.Env):
    """Heated serial-cascade control env over IA2 (or direct Modbus).

    observation = the 14 sensors in ia2_config.json order (engineering units):
                  3 levels, 3 temps, 3 flows, 5 digital safety flags.
    action      = 4 actuator fractions in [0,1] (v_12, v_23, e_101, vfd).
    reward      = -(level + temp tracking error vs setpoints + action cost).
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict | str | Path | None = None,
                 backend: str = "auto", control_dt: float = 0.5, mode: str = "rl",
                 port: int | None = None):
        super().__init__()
        self.config = load_config(config) if not isinstance(config, dict) else config
        self.control_dt = float(control_dt)
        self._port_override = port

        self.regs = _registers_by_address(self.config)
        self.reg_by_name = {r["name"]: r for r in self.regs}
        self.sensor_names = list(self.config["sensors"])
        self.actuator_names = list(self.config["actuators"])
        self.scales = {r["name"]: float(r["scale"]) for r in self.regs}  # uint16 raw regs only (reset)
        self.setpoints = {k: float(v) for k, v in
                          self.config["control"]["setpoints_m"].items()}
        self.temp_setpoints = {k: float(v) for k, v in
                               self.config["control"].get("setpoints_c", {}).items()}
        rw = self.config["control"].get("reward_weights", {})
        self.reward_weights = {
            "level": float(rw.get("level", 1.0)),
            "temp": float(rw.get("temp", 0.0001)),
            "action": float(rw.get("action", 0.01)),
        }
        # per-actuator engineering max (from the contract register max) for [0,1] -> eng scaling
        self._act_max = {n: float(self.reg_by_name[n]["max"]) for n in self.actuator_names}
        self._reset_nonce = 0

        self.backend: Backend = self._make_backend(backend)
        self.mode = mode.lower()
        valid_modes = {"manual", "pid", "mpc", "rl"}
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Please use one of: {', '.join(sorted(valid_modes))}")
        self._mode_int = {"manual": 0, "pid": 1, "mpc": 2, "rl": 3}[self.mode]
        self._write_names, self._write_max = self._write_targets()
        if self.backend.writes_via_plc:
            self.backend.write_register("mode", self._mode_int)  # PLC CASE selector
            LOG.info("mode = %s (%d)", self.mode, self._mode_int)

        sreg = self.reg_by_name
        obs_lo = np.array([float(sreg[n]["min"]) for n in self.sensor_names], dtype=np.float32)
        obs_hi = np.array([float(sreg[n]["max"]) for n in self.sensor_names], dtype=np.float32)
        self.observation_space = spaces.Box(obs_lo, obs_hi, dtype=np.float32)
        self.action_space = spaces.Box(
            np.zeros(len(self.actuator_names), np.float32),
            np.ones(len(self.actuator_names), np.float32), dtype=np.float32,
        )

    # ---- backend selection ----
    def _make_backend(self, kind: str) -> Backend:
        if kind.startswith("edge"):
            ia2 = self.config["ia2"]
            edge_name = kind.split(":", 1)[1] if ":" in kind else ia2.get("edge_name")
            if not edge_name:
                raise RuntimeError(
                    "--backend edge requires a name: use 'edge:<name>' or set "
                    "ia2.edge_name in ia2_config.json")
            be = EdgeBackend(ia2["server_url"], ia2.get("project_name"), edge_name)
            LOG.info("backend = Edge (%s, edge=%s)", ia2["server_url"], edge_name)
            return be
        if kind in ("ia2", "auto"):
            try:
                ia2 = self.config["ia2"]
                be = IA2Backend(ia2["server_url"], ia2.get("project_name"),
                                probe_runtime=(kind == "auto"))
                LOG.info("backend = IA2 (%s)", ia2["server_url"])
                return be
            except Exception as e:
                if kind == "ia2":
                    raise
                LOG.warning("IA2 backend unavailable (%s); using Modbus", e)
        m = self.config["modbus"]
        port = int(self._port_override) if self._port_override else int(m["port"])
        be = ModbusBackend(m["host"], port, self.regs)
        LOG.info("backend = Modbus (%s:%s, multi-slave)", m["host"], port)
        return be

    # ---- conversions ----
    def _decode_obs(self, raw: dict) -> np.ndarray:
        # raw values are already engineering units (floats decoded by the Modbus
        # backend or REAL vars from IA2); assemble sensors in config order.
        vals = []
        for name in self.sensor_names:
            if name not in raw:
                raise RuntimeError(
                    f"sensor '{name}' missing from backend read; got keys={list(raw)}")
            vals.append(float(raw[name]))
        return np.asarray(vals, dtype=np.float32)

    def _write_targets(self) -> tuple[list[str], list[float]]:
        """Var names + engineering maxima the agent writes each step, by mode/backend."""
        if not self.backend.writes_via_plc:                      # modbus -> drive cabinet directly
            return list(self.actuator_names), [self._act_max[n] for n in self.actuator_names]
        if self.mode == "manual":
            return (["manual_vfd", "manual_v12", "manual_v23", "manual_v33", "manual_h1"],
                    [50.0, 100.0, 100.0, 100.0, 100.0])
        if self.mode == "pid":
            return (["tank1_level_sp", "tank2_level_sp", "tank3_level_sp", "tank1_temp_sp"],
                    [0.5, 0.5, 0.5, 100.0])
        return ([f"{n}_req" for n in self.actuator_names],       # mpc / rl
                [self._act_max[n] for n in self.actuator_names])

    def _action_to_writes(self, action) -> dict:
        a = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        return {name: float(a[i] * mx)
                for i, (name, mx) in enumerate(zip(self._write_names, self._write_max))}

    def setpoint_action(self) -> np.ndarray:
        """Config setpoints as a normalized [0,1] action (PID-mode demo)."""
        sp = self.config["control"]
        return np.array([
            sp["setpoints_m"]["tank1_level"] / 0.5,
            sp["setpoints_m"]["tank2_level"] / 0.5,
            sp["setpoints_m"]["tank3_level"] / 0.5,
            sp["setpoints_c"]["tank1_temp"] / 100.0,
        ], dtype=np.float32)

    def _reward(self, action, obs) -> tuple[float, dict]:
        sidx = {n: i for i, n in enumerate(self.sensor_names)}
        levels = {n: float(obs[sidx[n]]) for n in self.setpoints}            # level sensors
        temps = {n: float(obs[sidx[n]]) for n in self.temp_setpoints}        # temp sensors
        w = self.reward_weights
        track_l = sum((levels[n] - self.setpoints[n]) ** 2 for n in levels)
        track_t = sum((temps[n] - self.temp_setpoints[n]) ** 2 for n in temps)
        clipped = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        action_cost = w["action"] * float(np.sum(clipped))
        reward = float(-(w["level"] * track_l + w["temp"] * track_t + action_cost))
        info = {
            "levels_m": levels, "temps_c": temps,
            "track_level_mse": track_l, "track_temp_mse": track_t,
            "action": np.asarray(action, dtype=np.float32).tolist(),
        }
        return reward, info

    # ---- Gym API ----
    def reset(self, *, seed=None, options=None):
        """Reset the plant to a sampled initial state (RL init-state distribution).

        Writes sampled initial levels to the init_h* registers (raw uint16 via
        scale), then pulses reset_cmd (slave 03). The cabinet snaps to init_h*
        and HOLDS while reset_cmd is asserted, so the obs read here are exactly
        the init levels. Releasing reset_cmd (-> 0) lets the cabinet resume.
        """
        super().reset(seed=seed)
        for name in self._write_names:                   # neutral the mode's write vars
            self.backend.write_register(name, 0.0)
        info: dict = {}
        rcfg = self.config.get("reset")
        if rcfg:
            lo, hi = rcfg.get("init_level_range_m", [0.10, 0.40])
            init_levels: dict[str, float] = {}
            for name in rcfg.get("init_levels", []):
                level = float(self.np_random.uniform(lo, hi))
                init_levels[name] = level
                raw = int(round(level / self.scales[name]))   # uint16 raw for init_h*
                self.backend.write_register(name, raw)
            info["init_levels_m"] = init_levels
            cmd = rcfg.get("command_register")
            if cmd:
                self._reset_nonce = self._reset_nonce % 65535 + 1  # never wraps to 0
                self.backend.write_register(cmd, self._reset_nonce)  # fresh nonce -> snap + hold
                time.sleep(self.control_dt)
                obs = self._decode_obs(self.backend.read_raw())
                self.backend.write_register(cmd, 0)                # release -> resume
                return obs, info
        time.sleep(self.control_dt)
        return self._decode_obs(self.backend.read_raw()), info

    def step(self, action):
        for name, value in self._action_to_writes(action).items():
            self.backend.write_register(name, value)
        time.sleep(self.control_dt)
        raw = self.backend.read_raw()
        self.last_raw = raw
        obs = self._decode_obs(raw)
        reward, info = self._reward(action, obs)
        info["raw"] = raw
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
    steps_data = []
    pid_act = env.setpoint_action() if mode == "pid" else None
    for k in range(steps):
        action = pid_act if pid_act is not None else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        steps_data.append({
            "step": k,
            "levels": [float(obs[0]), float(obs[2]), float(obs[4])],
            "temps": [float(obs[1]), float(obs[3]), float(obs[5])],
            "flows": [float(obs[6]), float(obs[7]), float(obs[8])],
            "action": [float(x) for x in action],
            "applied_duty": [float(info["raw"].get(n, 0.0)) / env._act_max[n]
                              for n in env.actuator_names],
            "reward": reward})
        if k % 4 == 0 or k == steps - 1:
            lv, tp = info["levels_m"], info.get("temps_c", {})
            LOG.info("step %3d  act=%s  levels(m)=%.3f/%.3f/%.3f  "
                     "temps(C)=%.1f/%.1f/%.1f  r=%.4f",
                     k, np.round(action, 2),
                     lv.get("tank1_level", float("nan")), lv.get("tank2_level", float("nan")),
                     lv.get("tank3_level", float("nan")),
                     tp.get("tank1_temp", float("nan")), tp.get("tank2_temp", float("nan")),
                     tp.get("tank3_temp", float("nan")), reward)
    env.close()
    LOG.info("rollout done — mean reward = %.4f over %d steps", float(np.mean(rewards)), steps)
    try:
        from controllers.rollout_report import report
        report(steps_data, tag=mode)
    except Exception as e:
        LOG.warning("rollout report skipped: %s", e)


def main():
    ap = argparse.ArgumentParser(description="AIO bridge Gym env (heated serial cascade).")
    ap.add_argument("--backend", default="auto",
                    help="auto | ia2 | modbus | edge | edge:<name> (default: auto)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--control-dt", type=float, default=0.5)
    ap.add_argument("--mode", default="rl",
                    choices=["manual", "pid", "mpc", "rl"],
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
