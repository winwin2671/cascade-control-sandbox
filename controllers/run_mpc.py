"""MPC supervisor — drives the live IA2 plant with the MPCAgent controller.

Sets PLC mode=MPC (the PLC passes actuator*_req through the L5 shield), then
loops: read the plant state from the IA2 snapshot -> MPCAgent.compute -> write
actuator*_req. The MPC predicts with ThreeTankModel (mirrors mock_cabinet).

Requires the IA2 chain up: mock_cabinet.py + ia2-server + `cs project open` +
`cs run` (see mpc_run.sh).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aio_bridge_env import CascadeBridgeEnv          # noqa: E402
from controllers.threetank_model import ThreeTankModel  # noqa: E402
from controllers.mpc_agent import MPCAgent            # noqa: E402

LOG = logging.getLogger("mpc_supervisor")


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("pymodbus").setLevel(logging.WARNING)

    env = CascadeBridgeEnv(backend="ia2", control_dt=0.5, mode="mpc")
    model = ThreeTankModel()
    mpc = MPCAgent(model, Ts=0.5, P=20)
    mpc.reset()

    cfg = env.config["control"]
    hsp, tsp = cfg["setpoints_m"], cfg["setpoints_c"]
    sp = {"h_sp": [hsp["tank1_level"], hsp["tank2_level"], hsp["tank3_level"]],
          "t_sp": [tsp["tank1_temp"], tsp["tank2_temp"], tsp["tank3_temp"]]}

    obs, _ = env.reset()
    LOG.info("MPC supervisor start — sp levels=%s m, temps=%s degC", sp["h_sp"], sp["t_sp"])
    rewards = []
    steps_data = []

    for k in range(40):
        # obs = [h1, T1, h2, T2, h3, T3] in engineering units
        meas = {"levels": [float(obs[0]), float(obs[2]), float(obs[4])],
                "temps": [float(obs[1]), float(obs[3]), float(obs[5])],
                "t_cold": model.t_supply, "t_amb": model.t_ambient}
        act = mpc.compute(meas, sp, env.control_dt)
        a = list(act["pumps"]) + list(act["valves"]) + list(act["heaters"])  # [p1,p2,h1,h2,h3]
        obs, reward, _, _, info = env.step(a)
        if k % 4 == 0 or k == 39:
            lv, tp = info["levels_m"], info.get("temps_c", {})
            LOG.info("step %2d  act=%s  levels(m)=%.3f/%.3f/%.3f  temps(C)=%.1f/%.1f/%.1f  r=%.3f",
                     k, [round(float(x), 2) for x in a],
                     lv.get("tank1_level", float("nan")),
                     lv.get("tank2_level", float("nan")),
                     lv.get("tank3_level", float("nan")),
                     tp.get("tank1_temp", float("nan")),
                     tp.get("tank2_temp", float("nan")),
                     tp.get("tank3_temp", float("nan")), reward)
        rewards.append(reward)
        steps_data.append({
            "step": k, "levels": [float(obs[0]), float(obs[2]), float(obs[4])],
            "temps": [float(obs[1]), float(obs[3]), float(obs[5])],
            "action": [float(x) for x in a], "reward": reward})
    env.close()
    LOG.info("rollout done — mean reward = %.4f over %d steps", np.mean(rewards), len(rewards))
    from controllers.rollout_report import report
    report(steps_data, tag="mpc")


if __name__ == "__main__":
    main()
