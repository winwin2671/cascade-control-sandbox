"""Register our 3-tank plant into AIO-Gym's module-level registries.

Calling `register_threetank()` makes `AIOGymNativeEnv("threetank", ...)` use our
ThreeTankModel (numpy physics mirroring mock_cabinet) + our ECON economics + the
PIDAgent gains/pairings, so AIO-Gym's env / trainers / scorer / evaluate run
unchanged against our plant. Runtime registry injection only — no AIO-Gym source
edit, no copy.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AIO_GYM = str(Path.home() / "projects" / "AIO-Gym")
for _p in (str(ROOT), AIO_GYM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import aiogym.models as _m      # noqa: E402
import aiogym.env as _e         # noqa: E402
import aiogym.baselines as _b   # noqa: E402
from controllers.threetank_model import ThreeTankModel  # noqa: E402

_registered = False


def register_threetank() -> None:
    """Idempotent: register ThreeTankModel + ECON + PID gains/pairings for the
    'threetank' scenario so AIOGymNativeEnv('threetank') works."""
    global _registered
    if _registered:
        return
    _m.MODELS["threetank"] = ThreeTankModel

    # ECON: mirror AIO-Gym's cascade structure (value="none" -> minimize heater
    # energy + band-violation penalties), with bands around OUR setpoints.
    _e.ECON["threetank"] = {
        "temp_band": [(40.0, 50.0), (40.0, 50.0), (40.0, 50.0)],   # +-5 degC around 45
        "level_band": [(0.35, 0.55), (0.35, 0.55)],                 # controlled tanks 0,2; +-0.1 m
        "value": "none", "w_value": 0.0, "w_energy": 0.7, "w_viol": 29.0,
    }

    # PIDAgent: pump0->tank1 level, pump1->tank3 level; heater0..2 -> tank1..3 temp.
    # Gains mirror cascade (tunable). demand_valve_index None (no valves, nV=0).
    _b.GAINS["threetank"] = {"level_pump": (8.0, 0.4, 0.0), "level_valve": (0.0, 0.0, 0.0),
                             "temp": (0.06, 0.01, 0.0)}
    _b.PAIRING["threetank"] = {"level": [("pump", 0, 0), ("pump", 1, 2)],
                               "temp": [(0, 0, False), (1, 1, False), (2, 2, False)],
                               "demand_valve_index": None, "holds": []}

    # Supervisory layout: RL picks temp setpoints (tank0..2) + level setpoints
    # (controlled tanks 0,2), PID tracks them. Enables action_mode="setpoint"
    # (AIO-Gym's default — RL-on-PID, much easier than direct actuator control).
    _e.SUPERVISORY["threetank"] = [
        ("t_sp", 0, 20.0, 80.0), ("t_sp", 1, 20.0, 80.0), ("t_sp", 2, 20.0, 80.0),
        ("h_sp", 0, 0.15, 0.55), ("h_sp", 2, 0.15, 0.55),
    ]
    # No PLANT_REGIME entry -> randomize_plant uses the default (no regime shift).
    _registered = True
