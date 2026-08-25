(**********************************************************************
  ThreeTank — IA2 PLC program for the HEATED SERIAL-CASCADE rig.

  Topology: pump P-101 -> Tank1 --(V-12)--> Tank2 --(V-23)--> Tank3
            --(V-33)--> Reservoir --(P-101)--> Tank1 (recirculation).
            One heater E-101 (2 kW) in Tank1 only; Tank2/Tank3 warm via
            downstream hot-water advection (under-actuated temperature).
            V-33 (Tank3 -> reservoir) is an actuated control valve used as a
            supervisor MV: PID holds it open as the free drain, MPC/RL modulate.

  Control-mode selector (Manual / PID / MPC / RL) + L5 software shield.
  On/off interlock-test solenoids SV-1..3 (DO coils, parallel to V-12/V-23/
  V-33) sit OUTSIDE the mode selector: test instrumentation only, gated by
  test_sv_en (see the SV section near the end of the body).

  Sensor channels are f32 (REAL POU vars, engineering units). Actuator channels
  are u16/FC06 holding registers, raw 0..10000 = 0..100% (MD200-style). So the
  mapped actuator vars are UINT 0..10000; the PID/MANSTATION work in REAL 0..100%
  and the L5 shield converts % -> raw via REAL_TO_INT(out * 100).

  `mode` (PLC-internal UINT, API-written; NOT a Modbus register) selects the
  request source per actuator each scan:
    0 MANUAL : upstream = manual_* (REAL %)
    1 PID    : upstream = FB_PID.out (REAL %)
    2 MPC    : upstream = *_cmd_req (REAL %)
    3 RL     : upstream = *_cmd_req (REAL %)
  FB_MANSTATION gives bumpless manual<->auto. The L5 shield then clamps +
  intercepts -> mapped UINT actuator, driven by the 5 FC02 hardware-safety flags.

  4 PID loops (engineering units): pump->tank1 level, V-12->tank2 level,
  V-23->tank3 level, E-101->tank1 temp (the only directly-heated tank).

  PID gains are PLACEHOLDERS for a physically-scaled 2 kW rig — tune at SAT.
***********************************************************************)
PROGRAM ThreeTank
  VAR
    (* sensors (iomap input, REAL via f32) *)
    tank1_level : REAL;  tank1_temp : REAL;
    tank2_level : REAL;  tank2_temp : REAL;
    tank3_level : REAL;  tank3_temp : REAL;
    tank1_flow  : REAL;  tank2_flow  : REAL;  tank3_flow  : REAL;
    (* DI safety statuses (iomap input, BOOL via FC02) *)
    di_dryfire         : BOOL;
    di_overflow        : BOOL;
    di_heater_contactor: BOOL;
    di_pump_contactor  : BOOL;
    di_estop           : BOOL;
    (* mapped actuators (iomap OUTPUT, UINT raw 0..10000; assigned ONLY here *)
    vfd_cmd   : UINT;   (* P-101 VFD speed, 0..10000 = 0..100.00% of max freq *)
    v_12_cmd  : UINT;   (* V-12 position, 0..10000 = 0..100% *)
    v_23_cmd  : UINT;   (* V-23 position, 0..10000 *)
    v_33_cmd  : UINT;   (* V-33 position, 0..10000 (Tank3 -> reservoir) *)
    e_101_cmd : UINT;   (* E-101 heater duty, 0..10000 *)
    sv_1_cmd  : BOOL;   (* SV-1 on/off solenoid coil (DO slave 07 ch 0), || V-12 *)
    sv_2_cmd  : BOOL;   (* SV-2 coil (ch 1), || V-23 *)
    sv_3_cmd  : BOOL;   (* SV-3 coil (ch 2), || V-33 *)
    (* interlock-test valve requests (PLC-internal, API-written; NOT in any
       control mode — test instrumentation is deliberately outside the agent
       action space). test_sv_en defaults FALSE -> coils forced closed. *)
    sv_1_req  : BOOL;
    sv_2_req  : BOOL;
    sv_3_req  : BOOL;
    test_sv_en: BOOL;
    (* agent requests / manual (PLC-internal, API-written, REAL 0..100%) *)
    vfd_cmd_req   : REAL;
    v_12_cmd_req  : REAL;
    v_23_cmd_req  : REAL;
    v_33_cmd_req  : REAL;
    e_101_cmd_req : REAL;
    (* episode reset (iomap output; env-driven; body does not assign) *)
    reset_cmd : UINT;
    init_h1   : UINT;
    init_h2   : UINT;
    init_h3   : UINT;
    (* control mode + setpoints + manual outs (PLC-internal, API-written) *)
    mode             : UINT;   (* 0=MANUAL 1=PID 2=MPC 3=RL *)
    tank1_level_sp   : REAL;   (* m *)
    tank2_level_sp   : REAL;
    tank3_level_sp   : REAL;
    tank1_temp_sp    : REAL;   (* degC (only Tank1 temp is PID-controlled) *)
    manual_vfd : REAL;         (* 0..100 % *)
    manual_v12 : REAL;
    manual_v23 : REAL;
    manual_v33 : REAL;
    manual_h1  : REAL;
    (* FB instances *)
    pid_l1 : FB_PID;  pid_l2 : FB_PID;  pid_l3 : FB_PID;  pid_t1 : FB_PID;
    msta_vfd : FB_MANSTATION;  msta_v12 : FB_MANSTATION;
    msta_v23 : FB_MANSTATION;  msta_v33 : FB_MANSTATION;  msta_h1  : FB_MANSTATION;
    (* per-actuator request working vars (REAL 0..100%; locals so FB outputs
       are never the leftmost operand of an IF — ironplc gotcha 8). *)
    up_vfd : REAL;  up_v12 : REAL;  up_v23 : REAL;  up_v33 : REAL;  up_h1 : REAL;
    out_vfd : REAL; out_v12 : REAL; out_v23 : REAL; out_v33 : REAL; out_h1 : REAL;
    (* R2 fix: independent interlock latches. Trip on DI/threshold, clear with margin. *)
    overflow_trip : BOOL;
    dryfire_trip  : BOOL;
    overtemp_trip : BOOL;
    (* Minor/#6: per-valve forced-drain latches + release dwells (anti-chatter) *)
    v12_force : BOOL;
    v23_force : BOOL;
    v33_force : BOOL;
    v12_dwell : INT;
    v23_dwell : INT;
    v33_dwell : INT;
    (* heartbeat *)
    scan_count : UDINT;
  END_VAR

  (* ---------- PID loops (sp/pv already engineering units via f32 iomap).
     Outputs are 0..100%; track_value reads the UINT actuator /100 -> %. ---------- *)
  pid_l1(pv := tank1_level, sp := tank1_level_sp,
         kp := 200.0, ti_s := 10.0, td_s := 0.0, dt_s := 0.05,
         direct := FALSE, auto := TRUE, manual_out := 0.0,
         out_min := 0.0, out_max := 100.0,           (* pump speed 0..100% *)
         track := (mode <> 1), track_value := INT_TO_REAL(vfd_cmd) * 1.0e-2);
  pid_l2(pv := tank2_level, sp := tank2_level_sp,
         kp := 200.0, ti_s := 10.0, td_s := 0.0, dt_s := 0.05,
         direct := FALSE, auto := TRUE, manual_out := 0.0,
         out_min := 0.0, out_max := 100.0,           (* V-12 0..100% *)
         track := (mode <> 1), track_value := INT_TO_REAL(v_12_cmd) * 1.0e-2);
  pid_l3(pv := tank3_level, sp := tank3_level_sp,
         kp := 200.0, ti_s := 10.0, td_s := 0.0, dt_s := 0.05,
         direct := FALSE, auto := TRUE, manual_out := 0.0,
         out_min := 0.0, out_max := 100.0,           (* V-23 0..100% *)
         track := (mode <> 1), track_value := INT_TO_REAL(v_23_cmd) * 1.0e-2);
  pid_t1(pv := tank1_temp, sp := tank1_temp_sp,
         kp := 2.0, ti_s := 60.0, td_s := 0.0, dt_s := 0.05,
         direct := FALSE, auto := TRUE, manual_out := 0.0,
         out_min := 0.0, out_max := 100.0,           (* E-101 heater 0..100% *)
         track := (mode <> 1), track_value := INT_TO_REAL(e_101_cmd) * 1.0e-2);

  (* ---------- mode selects the upstream request (REAL 0..100%) ---------- *)
  CASE mode OF
    0 (* MANUAL *):
      up_vfd := manual_vfd;  up_v12 := manual_v12;  up_v23 := manual_v23;
      up_v33 := manual_v33;  up_h1 := manual_h1;
    1 (* PID *):
      up_vfd := pid_l1.out;  up_v12 := pid_l2.out;  up_v23 := pid_l3.out;
      up_v33 := 100.0;       (* V-33 has no PID loop -> held open as the free drain *)
      up_h1 := pid_t1.out;
    2, 3 (* MPC, RL *):
      up_vfd := vfd_cmd_req;  up_v12 := v_12_cmd_req;  up_v23 := v_23_cmd_req;
      up_v33 := v_33_cmd_req;  up_h1 := e_101_cmd_req;
    ELSE  (* unknown mode -> safe (all outputs to 0) *)
      up_vfd := 0.0;  up_v12 := 0.0;  up_v23 := 0.0;  up_v33 := 0.0;  up_h1 := 0.0;
  END_CASE;

  (* ---------- FB_MANSTATION: manual/auto switch + clamp (all 0..100%) ----------
     balance := FALSE (corr always 0). With balance := (mode <= 1) the POU's
     boot sequence poisoned every station: the POU starts in mode 0 (MANUAL,
     outputs 0), then the supervisor writes mode=1 — that manual->auto EDGE
     back-calculated corr := out - auto_in and HELD it for the whole PID
     session. For V-33 (up_v33 = 100 vs manual 0) that meant corr = -100 and
     out_v33 = 0 forever: the free drain never opened, T3 froze above its
     setpoint, and every inflow PID above setpoint shut in a chain — the plant
     deadlocked with T1 cooking alone (measured: levels frozen 0.33/0.39/0.40,
     T1 -> 49 C, T2/T3 stuck at 25). The other four stations carried the same
     corr = -(output at the flip) bias, just small. Cost of FALSE: a manual<->
     PID mode switch is a step, not bumpless — acceptable while the supervisor
     owns mode changes; revisit (one-scan balance pulse) if an operator panel
     gets live mode authority. *)
  msta_vfd(auto_in := up_vfd, auto := (mode <> 0), manual_out := manual_vfd,
           out_min := 0.0, out_max := 100.0, balance := FALSE, dt_s := 0.05);
  msta_v12(auto_in := up_v12, auto := (mode <> 0), manual_out := manual_v12,
           out_min := 0.0, out_max := 100.0, balance := FALSE, dt_s := 0.05);
  msta_v23(auto_in := up_v23, auto := (mode <> 0), manual_out := manual_v23,
           out_min := 0.0, out_max := 100.0, balance := FALSE, dt_s := 0.05);
  msta_v33(auto_in := up_v33, auto := (mode <> 0), manual_out := manual_v33,
           out_min := 0.0, out_max := 100.0, balance := FALSE, dt_s := 0.05);
  msta_h1(auto_in := up_h1, auto := (mode <> 0), manual_out := manual_h1,
          out_min := 0.0, out_max := 100.0, balance := FALSE, dt_s := 0.05);
  out_vfd := msta_vfd.out;  out_v12 := msta_v12.out;  out_v23 := msta_v23.out;
  out_v33 := msta_v33.out;  out_h1 := msta_h1.out;

  (* ---------- L5 shield: 0..100% -> UINT 0..10000 (x100) + hardware/software interlocks.
     R2 fix: 3 independent latches (overflow / dry-fire / over-temp).
     Recovery rule: a trip may only inhibit the actuator that AGGRAVATES it —
     overflow cuts the pump (inflow) but never the drains, dry-fire/over-temp cut
     the heater. Cutting the recovery path would latch the trip forever.
     B1 fix (#6): di_estop is an NC chain — 0 = PRESSED, 1 = healthy — so the
     tripped condition is NOT di_estop. (The old TRUE-as-tripped reading latched
     every trip on the first scan of healthy hardware, and only PRESSING the
     E-stop could unlatch them — the safety function was inverted.) ---------- *)
  IF di_overflow OR NOT di_estop THEN overflow_trip := TRUE; END_IF;
  IF (tank1_level < 0.40) AND (tank2_level < 0.40) AND (tank3_level < 0.40)
     AND NOT di_overflow AND di_estop THEN overflow_trip := FALSE; END_IF;

  IF di_dryfire OR NOT di_estop THEN dryfire_trip := TRUE; END_IF;
  IF (tank1_level > 0.06) AND NOT di_dryfire AND di_estop THEN dryfire_trip := FALSE; END_IF;

  IF (tank1_temp > 70.0) OR NOT di_estop THEN overtemp_trip := TRUE; END_IF;
  IF (tank1_temp < 65.0) AND di_estop THEN overtemp_trip := FALSE; END_IF;

  (* pump VFD: cut on overflow / e-stop — stop INFLOW so the tank can drain *)
  IF overflow_trip THEN
    vfd_cmd := 0;
  ELSIF out_vfd > 100.0 THEN
    vfd_cmd := 10000;
  ELSE
    vfd_cmd := REAL_TO_INT(out_vfd * 100.0);
  END_IF;

  (* V-12: NOT cut on overflow — cutting the drains would sustain the overflow
     (pump off + valves shut = frozen level). While the latch is held and Tank1 is
     still above the recovery threshold, the drain is FORCED open — the pump
     is cut so this valve is the only recovery path, and leaving it to the
     controller (which may command it shut) would latch forever.
     Minor/#6: hysteresis (force above 0.40, release below 0.35) — a level riding
     the threshold with inflow otherwise toggles the forcing at scan rate
     (measured 0.05 s chatter; 129 trip/re-trip cycles in 900 s under a
     persistent refill). *)
  IF overflow_trip AND (tank1_level > 0.40) THEN v12_force := TRUE; END_IF;
  IF (NOT overflow_trip) OR (tank1_level < 0.35) THEN v12_force := FALSE; END_IF;
  IF v12_force OR (v12_dwell > 0) THEN
    v_12_cmd := 10000;
  ELSIF out_v12 > 100.0 THEN
    v_12_cmd := 10000;
  ELSE
    v_12_cmd := REAL_TO_INT(out_v12 * 100.0);
  END_IF;

  (* V-23: same pattern (hysteresis band + force latch) on Tank2 *)
  IF overflow_trip AND (tank2_level > 0.40) THEN v23_force := TRUE; END_IF;
  IF (NOT overflow_trip) OR (tank2_level < 0.35) THEN v23_force := FALSE; END_IF;
  IF v23_force OR (v23_dwell > 0) THEN
    v_23_cmd := 10000;
  ELSIF out_v23 > 100.0 THEN
    v_23_cmd := 10000;
  ELSE
    v_23_cmd := REAL_TO_INT(out_v23 * 100.0);
  END_IF;

  (* V-33 (Tank3 -> reservoir): same pattern on Tank3 *)
  IF overflow_trip AND (tank3_level > 0.40) THEN v33_force := TRUE; END_IF;
  IF (NOT overflow_trip) OR (tank3_level < 0.35) THEN v33_force := FALSE; END_IF;
  IF v33_force OR (v33_dwell > 0) THEN
    v_33_cmd := 10000;
  ELSIF out_v33 > 100.0 THEN
    v_33_cmd := 10000;
  ELSE
    v_33_cmd := REAL_TO_INT(out_v33 * 100.0);
  END_IF;

  (* E-101 heater: cut on dry-fire / over-temp / e-stop *)
  IF dryfire_trip OR overtemp_trip THEN
    e_101_cmd := 0;
  ELSIF out_h1 > 100.0 THEN
    e_101_cmd := 10000;
  ELSE
    e_101_cmd := REAL_TO_INT(out_h1 * 100.0);
  END_IF;

  (* Minor/#6: release dwell — after the force latch drops, keep the drain open
     for 20 more scans (1 s) so a level hovering at the 0.35 release threshold
     can't re-latch at scan rate. Reloaded to 20 while forced, counts down after. *)
  IF v12_force THEN v12_dwell := 20; ELSIF v12_dwell > 0 THEN v12_dwell := v12_dwell - 1; END_IF;
  IF v23_force THEN v23_dwell := 20; ELSIF v23_dwell > 0 THEN v23_dwell := v23_dwell - 1; END_IF;
  IF v33_force THEN v33_dwell := 20; ELSIF v33_dwell > 0 THEN v33_dwell := v33_dwell - 1; END_IF;

  (* ---------- On/off interlock-test solenoids SV-1..3 (DO slave 07, FC05 coils).
     Each is PARALLEL to its modulating valve (SV-1 || V-12, SV-2 || V-23,
     SV-3 || V-33) so energizing one gives full-bore flow on that path
     regardless of valve position — the scripted level transient for SAT
     interlock testing (drive a tank to LSH overflow or LSL dry-fire and watch
     the hardwired chain + this POU's L5 shield respond). Deliberately OUTSIDE
     the agent action space: no CASE-mode wiring, no PID/MPC/RL request path —
     only the supervisor/test harness writes sv_*_req. test_sv_en (default
     FALSE) gates all three coils so a stale request cannot hold a test valve
     open after the harness exits or across a POU restart. ---------- *)
  sv_1_cmd := sv_1_req AND test_sv_en;
  sv_2_cmd := sv_2_req AND test_sv_en;
  sv_3_cmd := sv_3_req AND test_sv_en;

  scan_count := scan_count + 1;
END_PROGRAM
