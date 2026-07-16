(**********************************************************************
  ThreeTank — IA2 PLC program for the 3-tank cascade sandbox.

  Control-mode selector (Manual / PID / MPC / RL) + L5 software shield.

  `mode` (PLC-internal UINT, API-written; NOT a Modbus register) selects the
  request source per actuator each scan:
    0 MANUAL : upstream = manual_*        (operator, via FB_MANSTATION manual)
    1 PID    : upstream = FB_PID.out      (in-PLC PID tracks a *_sp setpoint)
    2 MPC    : upstream = actuator*_req   (external supervisor, direct)
    3 RL     : upstream = actuator*_req   (external supervisor, direct)
  FB_MANSTATION (always in the chain) gives bumpless manual<->auto transfer.
  The L5 shield then clamps + intercepts the request -> mapped actuator:
    pumps OFF above 0.55 m (preempt L3), heaters OFF below 0.05 m (preempt L2
    dry-fire) or above 70 degC (preempt L4). L5 is mode-independent.

  All agent-writable vars are raw UINT (the IA2 variable-write API takes ints
  only); engineering conversion is done here. Sensors: level x1e-4 -> m,
  temp x1e-2 -> degC. Setpoints/manual/req: 0..10000 = 0..100 %, level_sp
  0..6000 = 0..0.6 m, temp_sp 0..10000 = 0..100 degC. Mapped actuators raw
  0..10000. 5 PID loops: pump1->tank1 level, pump2->tank3 level, heater1..3
  -> tank1..3 temp. PIDs run every scan; in non-PID modes they track the current
  actuator output (track := mode <> 1) -> bumpless return to PID.

  Agent writes (PLC-internal, API-written, NOT iomap-mapped -> persist): mode,
  the 5 *_sp setpoints, the 5 manual_*, and in MPC/RL the actuator*_req. This
  body is the sole writer of the mapped actuator*/heater* (iomap output).
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
    (* mapped actuators (iomap OUTPUT; assigned ONLY here, after the L5 shield. *)
    actuator1 : UINT;
    actuator2 : UINT;
    heater1   : UINT;
    heater2   : UINT;
    heater3   : UINT;
    (* agent requests for MPC/RL (NOT iomap-mapped; API-written; body never assigns). *)
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
    (* control mode + setpoints + manual outs (PLC-internal, API-written, raw). *)
    mode            : UINT;   (* 0=MANUAL 1=PID 2=MPC 3=RL *)
    tank1_level_sp  : UINT;   (* raw 0..6000 = 0..0.6 m *)
    tank3_level_sp  : UINT;
    tank1_temp_sp   : UINT;   (* raw 0..10000 = 0..100 degC *)
    tank2_temp_sp   : UINT;
    tank3_temp_sp   : UINT;
    manual_p1 : UINT;         (* raw 0..10000 = 0..100 % *)
    manual_p2 : UINT;
    manual_h1 : UINT;
    manual_h2 : UINT;
    manual_h3 : UINT;
    (* FB instances *)
    pid_level1 : FB_PID;
    pid_level3 : FB_PID;
    pid_temp1  : FB_PID;
    pid_temp2  : FB_PID;
    pid_temp3  : FB_PID;
    msta_p1 : FB_MANSTATION;
    msta_p2 : FB_MANSTATION;
    msta_h1 : FB_MANSTATION;
    msta_h2 : FB_MANSTATION;
    msta_h3 : FB_MANSTATION;
    (* per-actuator request working vars (REAL 0..100); locals so FB outputs are
       never the leftmost operand of an IF condition (ironplc gotcha 8). *)
    up_p1 : REAL;  up_p2 : REAL;
    up_h1 : REAL;  up_h2 : REAL;  up_h3 : REAL;
    out_p1 : REAL; out_p2 : REAL;
    out_h1 : REAL; out_h2 : REAL; out_h3 : REAL;
    (* C11: L5 interlock hysteresis latches (trip at threshold, clear with margin). *)
    lvl1_trip : BOOL;  lvl2_trip : BOOL;  lvl3_trip : BOOL;
    t1_trip : BOOL;    t2_trip : BOOL;    t3_trip : BOOL;
    (* heartbeat *)
    scan_count : UDINT;
  END_VAR

  (* ---------- PID loops (sp/pv scaled raw -> REAL eng) ---------- *)
  pid_level1(pv := INT_TO_REAL(tank1_level) * 1.0e-4, sp := INT_TO_REAL(tank1_level_sp) * 1.0e-4,
             kp := 500.0, ti_s := 10.0, td_s := 0.0, dt_s := 0.05,
             direct := FALSE, auto := TRUE, manual_out := 0.0,
             out_min := 0.0, out_max := 100.0,
             track := (mode <> 1), track_value := INT_TO_REAL(actuator1) * 1.0e-2);
  pid_level3(pv := INT_TO_REAL(tank3_level) * 1.0e-4, sp := INT_TO_REAL(tank3_level_sp) * 1.0e-4,
             kp := 500.0, ti_s := 10.0, td_s := 0.0, dt_s := 0.05,
             direct := FALSE, auto := TRUE, manual_out := 0.0,
             out_min := 0.0, out_max := 100.0,
             track := (mode <> 1), track_value := INT_TO_REAL(actuator2) * 1.0e-2);
  pid_temp1(pv := INT_TO_REAL(tank1_temp) * 1.0e-2, sp := INT_TO_REAL(tank1_temp_sp) * 1.0e-2,
            kp := 3.0, ti_s := 60.0, td_s := 0.0, dt_s := 0.05,
            direct := FALSE, auto := TRUE, manual_out := 0.0,
            out_min := 0.0, out_max := 100.0,
            track := (mode <> 1), track_value := INT_TO_REAL(heater1) * 1.0e-2);
  pid_temp2(pv := INT_TO_REAL(tank2_temp) * 1.0e-2, sp := INT_TO_REAL(tank2_temp_sp) * 1.0e-2,
            kp := 3.0, ti_s := 60.0, td_s := 0.0, dt_s := 0.05,
            direct := FALSE, auto := TRUE, manual_out := 0.0,
            out_min := 0.0, out_max := 100.0,
            track := (mode <> 1), track_value := INT_TO_REAL(heater2) * 1.0e-2);
  pid_temp3(pv := INT_TO_REAL(tank3_temp) * 1.0e-2, sp := INT_TO_REAL(tank3_temp_sp) * 1.0e-2,
            kp := 3.0, ti_s := 60.0, td_s := 0.0, dt_s := 0.05,
            direct := FALSE, auto := TRUE, manual_out := 0.0,
            out_min := 0.0, out_max := 100.0,
            track := (mode <> 1), track_value := INT_TO_REAL(heater3) * 1.0e-2);

  (* ---------- mode selects the upstream request (REAL 0..100) ---------- *)
  CASE mode OF
    0 (* MANUAL *):
      up_p1 := INT_TO_REAL(manual_p1) * 1.0e-2;
      up_p2 := INT_TO_REAL(manual_p2) * 1.0e-2;
      up_h1 := INT_TO_REAL(manual_h1) * 1.0e-2;
      up_h2 := INT_TO_REAL(manual_h2) * 1.0e-2;
      up_h3 := INT_TO_REAL(manual_h3) * 1.0e-2;
    1 (* PID *):
      up_p1 := pid_level1.out; up_p2 := pid_level3.out;
      up_h1 := pid_temp1.out;  up_h2 := pid_temp2.out;  up_h3 := pid_temp3.out;
    2, 3 (* MPC, RL *):
      up_p1 := INT_TO_REAL(actuator1_req) * 1.0e-2;
      up_p2 := INT_TO_REAL(actuator2_req) * 1.0e-2;
      up_h1 := INT_TO_REAL(heater1_req) * 1.0e-2;
      up_h2 := INT_TO_REAL(heater2_req) * 1.0e-2;
      up_h3 := INT_TO_REAL(heater3_req) * 1.0e-2;
    ELSE  (* C12: unknown mode -> safe (all outputs to 0) *)
      up_p1 := 0.0; up_p2 := 0.0;
      up_h1 := 0.0; up_h2 := 0.0; up_h3 := 0.0;
  END_CASE;

  (* ---------- FB_MANSTATION: bumpless manual<->auto (manual_out raw->REAL) ---------- *)
  msta_p1(auto_in := up_p1, auto := (mode <> 0), manual_out := INT_TO_REAL(manual_p1) * 1.0e-2,
          out_min := 0.0, out_max := 100.0, balance := (mode <= 1), dt_s := 0.05);
  msta_p2(auto_in := up_p2, auto := (mode <> 0), manual_out := INT_TO_REAL(manual_p2) * 1.0e-2,
          out_min := 0.0, out_max := 100.0, balance := (mode <= 1), dt_s := 0.05);
  msta_h1(auto_in := up_h1, auto := (mode <> 0), manual_out := INT_TO_REAL(manual_h1) * 1.0e-2,
          out_min := 0.0, out_max := 100.0, balance := (mode <= 1), dt_s := 0.05);
  msta_h2(auto_in := up_h2, auto := (mode <> 0), manual_out := INT_TO_REAL(manual_h2) * 1.0e-2,
          out_min := 0.0, out_max := 100.0, balance := (mode <= 1), dt_s := 0.05);
  msta_h3(auto_in := up_h3, auto := (mode <> 0), manual_out := INT_TO_REAL(manual_h3) * 1.0e-2,
          out_min := 0.0, out_max := 100.0, balance := (mode <= 1), dt_s := 0.05);
  out_p1 := msta_p1.out; out_p2 := msta_p2.out;
  out_h1 := msta_h1.out; out_h2 := msta_h2.out; out_h3 := msta_h3.out;

  (* ---------- L5 shield: eng (0..100) -> raw (0..10000) + cutoffs ---------- *)
  (* C11: hysteresis — trip at the threshold, clear with a margin to prevent
     actuator chatter at the boundary (500-count deadband = ~0.05 m / ~5 degC). *)
  IF (tank1_level > 5500) OR (tank2_level > 5500) THEN lvl1_trip := TRUE; END_IF;
  IF (tank1_level < 5000) AND (tank2_level < 5000) THEN lvl1_trip := FALSE; END_IF;
  IF lvl1_trip THEN
    actuator1 := 0;
  ELSIF out_p1 > 100.0 THEN
    actuator1 := 10000;
  ELSE
    actuator1 := REAL_TO_INT(out_p1 * 100.0);
  END_IF;

  IF (tank3_level > 5500) OR (tank2_level > 5500) THEN lvl3_trip := TRUE; END_IF;
  IF (tank3_level < 5000) AND (tank2_level < 5000) THEN lvl3_trip := FALSE; END_IF;
  IF lvl3_trip THEN
    actuator2 := 0;
  ELSIF out_p2 > 100.0 THEN
    actuator2 := 10000;
  ELSE
    actuator2 := REAL_TO_INT(out_p2 * 100.0);
  END_IF;

  IF (tank1_level < 500) THEN lvl1_trip := TRUE; END_IF;
  IF (tank1_level > 600) THEN lvl1_trip := FALSE; END_IF;
  IF (tank1_temp > 7000) THEN t1_trip := TRUE; END_IF;
  IF (tank1_temp < 6500) THEN t1_trip := FALSE; END_IF;
  IF lvl1_trip OR t1_trip THEN
    heater1 := 0;
  ELSIF out_h1 > 100.0 THEN
    heater1 := 10000;
  ELSE
    heater1 := REAL_TO_INT(out_h1 * 100.0);
  END_IF;

  IF (tank2_level < 500) THEN lvl2_trip := TRUE; END_IF;
  IF (tank2_level > 600) THEN lvl2_trip := FALSE; END_IF;
  IF (tank2_temp > 7000) THEN t2_trip := TRUE; END_IF;
  IF (tank2_temp < 6500) THEN t2_trip := FALSE; END_IF;
  IF lvl2_trip OR t2_trip THEN
    heater2 := 0;
  ELSIF out_h2 > 100.0 THEN
    heater2 := 10000;
  ELSE
    heater2 := REAL_TO_INT(out_h2 * 100.0);
  END_IF;

  IF (tank3_level < 500) THEN lvl3_trip := TRUE; END_IF;
  IF (tank3_level > 600) THEN lvl3_trip := FALSE; END_IF;
  IF (tank3_temp > 7000) THEN t3_trip := TRUE; END_IF;
  IF (tank3_temp < 6500) THEN t3_trip := FALSE; END_IF;
  IF lvl3_trip OR t3_trip THEN
    heater3 := 0;
  ELSIF out_h3 > 100.0 THEN
    heater3 := 10000;
  ELSE
    heater3 := REAL_TO_INT(out_h3 * 100.0);
  END_IF;

  scan_count := scan_count + 1;
END_PROGRAM
