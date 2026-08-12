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


def find_aiogym() -> str:
    """Resolve the AIO-Gym repo path. Checks: $AIO_GYM_PATH → sibling clone
    (../AIO-Gym, per the README setup) → ~/projects/AIO-Gym (WSL dev default)."""
    import os
    for candidate in [
        os.environ.get("AIO_GYM_PATH"),
        str(ROOT.parent / "AIO-Gym"),
        str(Path.home() / "projects" / "AIO-Gym"),
    ]:
        if candidate and Path(candidate, "aiogym", "__init__.py").exists():
            return candidate
    return str(Path.home() / "projects" / "AIO-Gym")  # fallback (error surfaces on import)


AIO_GYM = find_aiogym()
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

    # ECON: economic reward (value="none" -> minimize heater energy + band-violation
    # penalties), bands around OUR setpoints (0.30 m, 45 C).
    _e.ECON["threetank"] = {
        "temp_band": [(40.0, 50.0), (40.0, 50.0), (40.0, 50.0)],   # +-5 degC around 45
        "level_band": [(0.22, 0.40), (0.22, 0.40), (0.22, 0.40)],   # controlled tanks 0,1,2; +-0.08 m
        "value": "none", "w_value": 0.0, "w_energy": 0.7, "w_viol": 29.0,
    }

    # PIDAgent pairing (inflow-control): pump->tank1 level, valve0(V-12)->tank2,
    # valve1(V-23)->tank3 level; heater0(E-101)->tank1 temp only (T2/T3 warm via
    # advection, unactuated). V-33 (valve2, Tank3 drain) is a supervisor-only MV
    # (not PID-paired). Gains are placeholders — tune at SAT.
    _b.GAINS["threetank"] = {"level_pump": (8.0, 0.4, 0.0), "level_valve": (5.0, 0.25, 0.0),
                             "temp": (0.06, 0.01, 0.0)}
    _b.PAIRING["threetank"] = {"level": [("pump", 0, 0), ("valve", 0, 1), ("valve", 1, 2)],
                               "temp": [(0, 0, False)],
                               "demand_valve_index": None, "holds": []}

    # Supervisory layout: RL picks 3 temp setpoints + 3 level setpoints (PID tracks
    # them). Only T1 temp is directly reachable; T2/T3 targets shape the cascade.
    _e.SUPERVISORY["threetank"] = [
        ("t_sp", 0, 20.0, 80.0), ("t_sp", 1, 20.0, 80.0), ("t_sp", 2, 20.0, 80.0),
        ("h_sp", 0, 0.15, 0.45), ("h_sp", 1, 0.15, 0.45), ("h_sp", 2, 0.15, 0.45),
    ]
    # No PLANT_REGIME entry -> randomize_plant uses the default (no regime shift).
    _registered = True
