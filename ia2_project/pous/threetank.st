(**********************************************************************
  ThreeTank — IA2 PLC program for the 3-tank cascade sandbox.

  L5 SOFTWARE SHIELD (exploration safety).

  Per the project's 5-layer safety model, L1-L4 are HARDWARE — 30 mA RCD (L1),
  low-level switch / dry-fire (L2), high-level switch + passive standpipe (L3),
  heater thermal protector (L4) — and are the real, PLC-independent backstop.
  They are NOT modeled here. This body is L5: it intercepts the agent's actuator
  REQUESTS (the *_req vars, written via the runtime API) and, when an action
  would soon trip a hardware interlock, overrides it with the safe fallback
  (drive OFF). That keeps early random RL exploration from constantly tripping
  L2/L3/L4 (each of which needs a manual reset). L5 is an efficiency layer, NOT
  the safety backstop itself; unattended operation still requires L1-L4.

  Each scan, per output: clamp the request to 10000, then
    pumps  (actuator1 -> Tank1, actuator2 -> Tank3): OFF if that tank OR the
           downstream Tank2 is above 5500             (preempt L3 overflow);
    heaters (heater1..3): OFF if their tank is below 500 (preempt L2 dry-fire)
           OR above 7000                              (preempt L4 overtemp).

  Threshold values, raw uint16 (level x1e-4 -> m, temp x1e-2 -> degC):
    5500 = 0.55 m level cutoff (h_max 0.60 m, under L3 trip)
     500 = 0.05 m heater dry-fire threshold (L2)
    7000 = 70 degC temp cutoff (under the L4 element protector)
    10000 = raw full-scale pump/heater drive (ACT_MAX)
  Each sits BELOW the corresponding hardware trip so L5 preempts it.

  Data flow: the agent writes the *_req vars (this body never assigns them, so
  API writes persist). This body is the SOLE writer of the mapped actuator*/heater*
  vars, which the iomap output mappings forward to the cabinet registers.

  (Note: thresholds are inlined as literals because the vendored ironplc does
  not accept VAR CONSTANT or := initializers on regular VAR. For review:
  named constants + FB_LIMIT / FB_INTERLOCK8 from library/process-control/.)
***********************************************************************)
PROGRAM ThreeTank
  VAR
    (* sensors (iomap input, raw uint16) *)
    tank1_level : UINT;
    tank1_temp  : UINT;
    tank2_level : UINT;
    tank2_temp  : UINT;
    tank3_level : UINT;
    tank3_temp  : UINT;
    (* mapped actuators (iomap OUTPUT; assigned ONLY here, after the L5 checks.
       The agent never writes these directly.) *)
    actuator1 : UINT;
    actuator2 : UINT;
    heater1   : UINT;
    heater2   : UINT;
    heater3   : UINT;
    (* agent requests (NOT iomap-mapped; written between scans via the runtime
       variable-write API. This body never assigns them -> API writes persist.) *)
    actuator1_req : UINT;
    actuator2_req : UINT;
    heater1_req   : UINT;
    heater2_req   : UINT;
    heater3_req   : UINT;
    (* episode reset (iomap output; env-driven; body does not assign) *)
    reset_cmd : UINT;
    init_h1   : UINT;
    init_h2   : UINT;
    init_h3   : UINT;
    (* heartbeat *)
    scan_count : UDINT;
  END_VAR

  (* ---------- pumps: high-level cutoff (preempt L3 overflow) ---------- *)
  IF (tank1_level > 5500) OR (tank2_level > 5500) THEN
    actuator1 := 0;
  ELSIF actuator1_req > 10000 THEN
    actuator1 := 10000;
  ELSE
    actuator1 := actuator1_req;
  END_IF;

  IF (tank3_level > 5500) OR (tank2_level > 5500) THEN
    actuator2 := 0;
  ELSIF actuator2_req > 10000 THEN
    actuator2 := 10000;
  ELSE
    actuator2 := actuator2_req;
  END_IF;

  (* ---------- heaters: low-level dry-fire + high-temp cutoff
                (preempt L2 dry-fire and L4 overtemp) ---------- *)
  IF (tank1_level < 500) OR (tank1_temp > 7000) THEN
    heater1 := 0;
  ELSIF heater1_req > 10000 THEN
    heater1 := 10000;
  ELSE
    heater1 := heater1_req;
  END_IF;

  IF (tank2_level < 500) OR (tank2_temp > 7000) THEN
    heater2 := 0;
  ELSIF heater2_req > 10000 THEN
    heater2 := 10000;
  ELSE
    heater2 := heater2_req;
  END_IF;

  IF (tank3_level < 500) OR (tank3_temp > 7000) THEN
    heater3 := 0;
  ELSIF heater3_req > 10000 THEN
    heater3 := 10000;
  ELSE
    heater3 := heater3_req;
  END_IF;

  scan_count := scan_count + 1;
END_PROGRAM
