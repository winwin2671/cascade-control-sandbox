"""NMPC supervisor — drives the live IA2 plant with the CasADi+IPOPT NMPC oracle.

Sets PLC mode=MPC, then loops: read the plant state -> OracleAgent.compute ->
write actuator*_req. The NMPC predicts with the CasADi symbolic plant
(_f_threetank mirrors mock_cabinet). Slower per step than the numpy MPC (IPOPT
solve), but a true nonlinear optimizer.

Requires the IA2 chain up: mock_cabinet.py + ia2-server + `cs project open` +
`cs run` (see nmpc_run.sh).
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aio_bridge_env import CascadeBridgeEnv             # noqa: E402
from controllers.nmpc_oracle import OracleAgent          # noqa: E402

LOG = logging.getLogger("nmpc_supervisor")


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("pymodbus").setLevel(logging.WARNING)

    env = CascadeBridgeEnv(backend="ia2", control_dt=0.5, mode="mpc")
    # NMPC prediction dt matches the actual loop time (IPOPT solve ~1-2 s + 0.5 s
    # sleep ≈ 2 s/step); using 0.5 s here under-predicts the plant evolution
    # (the plant runs during the solve) and the levels overshoot. Shorter horizon
    # (N=10) keeps the solve fast enough to stay near real-time.
    agent = OracleAgent(horizon=10, control_dt=2.0)
    agent.reset()
    m = agent.model

    cfg = env.config["control"]
    hsp, tsp = cfg["setpoints_m"], cfg["setpoints_c"]
    sp = {"h_sp": [hsp["tank1_level"], hsp["tank2_level"], hsp["tank3_level"]],
          "t_sp": [tsp["tank1_temp"], tsp["tank2_temp"], tsp["tank3_temp"]]}

    obs, _ = env.reset()
    LOG.info("NMPC supervisor start — sp levels=%s m, temps=%s degC", sp["h_sp"], sp["t_sp"])

    for k in range(40):
        meas = {"levels": [float(obs[0]), float(obs[2]), float(obs[4])],
                "temps": [float(obs[1]), float(obs[3]), float(obs[5])],
                "t_cold": m.t_supply, "t_amb": m.t_ambient}
        t0 = time.time()
        act = agent.compute(meas, sp, env.control_dt)
        solve_s = time.time() - t0
        a = list(act["pumps"]) + list(act["valves"]) + list(act["heaters"])
        obs, reward, _, _, info = env.step(a)
        if k % 4 == 0 or k == 39:
            lv, tp = info["levels_m"], info.get("temps_c", {})
            LOG.info("step %2d  solve=%.2fs  act=%s  levels(m)=%.3f/%.3f/%.3f  "
                     "temps(C)=%.1f/%.1f/%.1f  r=%.3f",
                     k, solve_s, [round(float(x), 2) for x in a],
                     lv.get("tank1_level", float("nan")),
                     lv.get("tank2_level", float("nan")),
                     lv.get("tank3_level", float("nan")),
                     tp.get("tank1_temp", float("nan")),
                     tp.get("tank2_temp", float("nan")),
                     tp.get("tank3_temp", float("nan")), reward)
    env.close()


if __name__ == "__main__":
    main()
