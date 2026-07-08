(**********************************************************************
  ThreeTank — IA2 PLC program for the 3-tank cascade sandbox.

  Sensors (tankX_level / tankX_temp, UINT) are populated each scan by the
  iomap input mappings from the mock_cabinet Modbus registers (40001-40006).

  Actuators (actuator1 / actuator2, UINT, 0..10000 = pump drive fraction)
  are written between scans by the RL agent through aio_bridge_env.py, which
  calls the IA2 runtime write API (POST /api/runtime/variables/{name}). The
  iomap output mappings forward them to the cabinet registers (40007-40008).
  => This body must NOT overwrite actuator1/actuator2.

  All values are raw uint16; engineering conversion lives in ia2_config.json.
***********************************************************************)
PROGRAM ThreeTank
  VAR
    (* sensors (iomap input) *)
    tank1_level : UINT;
    tank1_temp  : UINT;
    tank2_level : UINT;
    tank2_temp  : UINT;
    tank3_level : UINT;
    tank3_temp  : UINT;
    (* actuators (iomap output; driven externally by the RL agent) *)
    actuator1 : UINT;
    actuator2 : UINT;
    (* heartbeat *)
    scan_count : UDINT;
  END_VAR

  scan_count := scan_count + 1;
END_PROGRAM
