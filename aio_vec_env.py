"""Vectorized training env — N cabinets on N ports, faster-than-real-time (G3/G5).

The RL TRAINING track. Spawns N mock_cabinet subprocesses (each --time-scale k on
a distinct port 5020+i) and wraps N Modbus-backend CascadeBridgeEnvs in a
Gymnasium AsyncVectorEnv — one worker process per env, so the per-step sleeps
overlap and you get real N x throughput on top of the time-scale (SyncVectorEnv
would serialize the sleeps and give no speedup). The IA2 validation track stays
single-instance (one PROGRAM per IA2 server — G5). With time-scale k and N envs,
throughput is ~kN x real-time.

The env's wall-clock step is `plant_dt / time_scale`, so each step is `plant_dt`
of PLANT time (the same control interval as real-time deployment) — but k x more
steps fit in a wall-second, and N envs run in parallel.

Usage:
    env, pool = make_vec_env(n=4, time_scale=10)
    obs, _ = env.reset()
    for step in range(1000):
        obs, reward, term, trunc, info = env.step(policy(obs))
    env.close(); pool.close()           # `with CabinetPool(...) as pool:` also works
"""
from __future__ import annotations

import atexit
import logging
import subprocess
import sys
import time
from pathlib import Path

from gymnasium.vector import AsyncVectorEnv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from aio_bridge_env import CascadeBridgeEnv  # noqa: E402

LOG = logging.getLogger("aio_vec_env")


class CabinetPool:
    """Spawn / hold / tear down N mock_cabinet subprocesses on distinct ports."""

    def __init__(self, n: int, time_scale: float = 1.0,
                 base_port: int = 5200, host: str = "127.0.0.1"):
        self.n = n
        self.time_scale = time_scale
        self.base_port = base_port
        self.host = host
        self.procs: list[subprocess.Popen] = []

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    def start(self) -> "CabinetPool":
        cab = str(ROOT / "mock_cabinet.py")
        for i in range(self.n):
            port = self.base_port + i
            self.procs.append(subprocess.Popen(
                [sys.executable, "-u", cab,
                 "--host", self.host, "--port", str(port),
                 "--time-scale", str(self.time_scale), "--log-every", "0"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        self._wait_ready()
        return self

    def _wait_ready(self, timeout: float = 20.0) -> None:
        from pymodbus.client import ModbusTcpClient
        deadline = time.time() + timeout
        for i in range(self.n):
            # C3 fix: verify the child process is alive before connecting — if it
            # died (e.g. bind failure from a stale cabinet on the port), abort early
            # instead of silently attaching to a foreign cabinet.
            if self.procs[i].poll() is not None:
                self.close()
                raise RuntimeError(
                    f"cabinet {i} (pid {self.procs[i].pid}) exited immediately — "
                    f"port {self.base_port + i} likely in use by a stale process")
            cl = ModbusTcpClient(host=self.host, port=self.base_port + i)
            ok = False
            while time.time() < deadline:
                if self.procs[i].poll() is not None:
                    self.close()
                    raise RuntimeError(
                        f"cabinet {i} (pid {self.procs[i].pid}) died during startup "
                        f"on port {self.base_port + i}")
                if cl.connect():
                    ok = True
                    break
                time.sleep(0.2)
            cl.close()
            if not ok:
                self.close()
                raise RuntimeError(
                    f"cabinet {i} (port {self.base_port + i}) did not bind in {timeout}s")
        LOG.info("%d cabinets ready (ports %d-%d, %sx)",
                 self.n, self.base_port, self.base_port + self.n - 1, self.time_scale)

    def close(self) -> None:
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=3)
            except Exception:
                p.kill()
        self.procs = []


def make_vec_env(n: int = 4, time_scale: float = 1.0,
                 base_port: int = 5020, plant_dt: float = 0.5
                 ) -> tuple[AsyncVectorEnv, CabinetPool]:
    """Create N cabinets + an AsyncVectorEnv over them.

    Returns (env, pool). The caller must `env.close()` and `pool.close()` (or use
    `with CabinetPool(...) as pool:`) to tear down the cabinet subprocesses.
    """
    pool = CabinetPool(n, time_scale, base_port).start()
    atexit.register(pool.close)
    wall_dt = plant_dt / time_scale if time_scale > 0 else plant_dt

    def make(i: int):
        return CascadeBridgeEnv(backend="modbus", port=base_port + i, control_dt=wall_dt)

    env = AsyncVectorEnv([lambda i=i: make(i) for i in range(n)])
    return env, pool
