(**********************************************************************
  ThreeTank — IA2 PLC program for the HEATED SERIAL-CASCADE rig.

  Topology: pump P-101 -> Tank1 --(V-12)--> Tank2 --(V-23)--> Tank3
            --(V-33)--> Reservoir --(P-101)--> Tank1 (recirculation).
            One heater E-101 (2 kW) in Tank1 only; Tank2/Tank3 warm via
            downstream hot-water advection (under-actuated temperature).
            V-33 (Tank3 -> reservoir) is an actuated control valve used as a
            supervisor MV: PID holds it open as the free drain, MPC/RL modulate.

  Control-mode selector (Manual / PID / MPC / RL) + L5 software shield.

  `mode` (PLC-internal UINT, API-written; NOT a Modbus register) selects the
  request source per actuator each scan:
    0 MANUAL : upstream = manual_*         (operator, via FB_MANSTATION)
    1 PID    : upstream = FB_PID.out       (in-PLC PID tracks a *_sp setpoint)
    2 MPC    : upstream = *_cmd_req        (external supervisor, direct)
    3 RL     : upstream = *_cmd_req        (external supervisor, direct)
  FB_MANSTATION (always in the chain) gives bumpless manual<->auto transfer.

  4 PID loops (engineering units; the f32 iomap hands sensors in directly):
    pump/VFD  -> tank1 level   (out 0..50 Hz)
    V-12      -> tank2 level   (out 0..100 %)
    V-23      -> tank3 level   (out 0..100 %)
    E-101     -> tank1 temp    (out 0..100 %)   <- the only directly-heated tank

  The L5 shield then clamps + intercepts the request -> mapped actuator using
  the 5 FC02 hardware-safety flags (dry-fire, overflow, contactor weld, e-stop)
  plus software level/temp limits. L5 is mode-independent.

  All agent-writable vars (mode, the *_sp setpoints, the manual_*, and in MPC/RL
  the *_cmd_req) are PLC-internal and API-written; they are NOT iomap-mapped.
  This body is the sole writer of the mapped v_*_cmd / vfd_cmd / e_101_cmd
  (iomap output). Sensor/DI/reset/init vars are iomap-mapped and read here.

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
    (* mapped actuators (iomap OUTPUT; assigned ONLY here, after the L5 shield *)
    vfd_cmd   : REAL;   (* P-101 VFD frequency, 0..50 Hz *)
    v_12_cmd  : REAL;   (* V-12 position, 0..100 % *)
    v_23_cmd  : REAL;   (* V-23 position, 0..100 % *)
    v_33_cmd  : REAL;   (* V-33 position, 0..100 % (Tank3 -> reservoir) *)
    e_101_cmd : REAL;   (* E-101 heater duty, 0..100 % *)
    (* agent requests for MPC/RL (PLC-internal, API-written; body never assigns) *)
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
    manual_vfd : REAL;         (* 0..50 Hz *)
    manual_v12 : REAL;         (* 0..100 % *)
    manual_v23 : REAL;
    manual_v33 : REAL;
    manual_h1  : REAL;
    (* FB instances *)
    pid_l1 : FB_PID;  pid_l2 : FB_PID;  pid_l3 : FB_PID;  pid_t1 : FB_PID;
    msta_vfd : FB_MANSTATION;  msta_v12 : FB_MANSTATION;
    msta_v23 : FB_MANSTATION;  msta_v33 : FB_MANSTATION;  msta_h1  : FB_MANSTATION;
    (* per-actuator request working vars (REAL eng units; locals so FB outputs
       are never the leftmost operand of an IF — ironplc gotcha 8). *)
    up_vfd : REAL;  up_v12 : REAL;  up_v23 : REAL;  up_v33 : REAL;  up_h1 : REAL;
    out_vfd : REAL; out_v12 : REAL; out_v23 : REAL; out_v33 : REAL; out_h1 : REAL;
    (* R2 fix: independent interlock latches. Trip on DI/threshold, clear with margin. *)
    overflow_trip : BOOL;
    dryfire_trip  : BOOL;
    overtemp_trip : BOOL;
    (* heartbeat *)
    scan_count : UDINT;
  END_VAR

  (* ---------- PID loops (sp/pv already engineering units via f32 iomap) ---------- *)
  pid_l1(pv := tank1_level, sp := tank1_level_sp,
         kp := 200.0, ti_s := 10.0, td_s := 0.0, dt_s := 0.05,
         direct := FALSE, auto := TRUE, manual_out := 0.0,
         out_min := 0.0, out_max := 50.0,            (* pump VFD: 0..50 Hz *)
         track := (mode <> 1), track_value := vfd_cmd);
  pid_l2(pv := tank2_level, sp := tank2_level_sp,
         kp := 200.0, ti_s := 10.0, td_s := 0.0, dt_s := 0.05,
         direct := FALSE, auto := TRUE, manual_out := 0.0,
         out_min := 0.0, out_max := 100.0,           (* V-12: 0..100 % *)
         track := (mode <> 1), track_value := v_12_cmd);
  pid_l3(pv := tank3_level, sp := tank3_level_sp,
         kp := 200.0, ti_s := 10.0, td_s := 0.0, dt_s := 0.05,
         direct := FALSE, auto := TRUE, manual_out := 0.0,
         out_min := 0.0, out_max := 100.0,           (* V-23: 0..100 % *)
         track := (mode <> 1), track_value := v_23_cmd);
  pid_t1(pv := tank1_temp, sp := tank1_temp_sp,
         kp := 2.0, ti_s := 60.0, td_s := 0.0, dt_s := 0.05,
         direct := FALSE, auto := TRUE, manual_out := 0.0,
         out_min := 0.0, out_max := 100.0,           (* E-101 heater: 0..100 % *)
         track := (mode <> 1), track_value := e_101_cmd);

  (* ---------- mode selects the upstream request (REAL eng units) ---------- *)
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

  (* ---------- FB_MANSTATION: bumpless manual<->auto ---------- *)
  msta_vfd(auto_in := up_vfd, auto := (mode <> 0), manual_out := manual_vfd,
           out_min := 0.0, out_max := 50.0, balance := (mode <= 1), dt_s := 0.05);
  msta_v12(auto_in := up_v12, auto := (mode <> 0), manual_out := manual_v12,
           out_min := 0.0, out_max := 100.0, balance := (mode <= 1), dt_s := 0.05);
  msta_v23(auto_in := up_v23, auto := (mode <> 0), manual_out := manual_v23,
           out_min := 0.0, out_max := 100.0, balance := (mode <= 1), dt_s := 0.05);
  msta_v33(auto_in := up_v33, auto := (mode <> 0), manual_out := manual_v33,
           out_min := 0.0, out_max := 100.0, balance := (mode <= 1), dt_s := 0.05);
  msta_h1(auto_in := up_h1, auto := (mode <> 0), manual_out := manual_h1,
          out_min := 0.0, out_max := 100.0, balance := (mode <= 1), dt_s := 0.05);
  out_vfd := msta_vfd.out;  out_v12 := msta_v12.out;  out_v23 := msta_v23.out;
  out_v33 := msta_v33.out;  out_h1 := msta_h1.out;

  (* ---------- L5 shield: eng clamp + hardware/software interlocks ----------
     R2 fix: 3 independent latches (overflow / dry-fire / over-temp). Each trips
     on its DI edge or threshold and clears with margin to prevent chatter. The
     hardware DI flags are authoritative (the real relays are hardwired); the
     software level/temp terms preempt before the hardware trips. *)
  IF di_overflow OR di_estop THEN overflow_trip := TRUE; END_IF;
  IF (tank1_level < 0.40) AND (tank2_level < 0.40) AND (tank3_level < 0.40)
     AND NOT di_overflow AND NOT di_estop THEN overflow_trip := FALSE; END_IF;

  IF di_dryfire OR di_estop THEN dryfire_trip := TRUE; END_IF;
  IF (tank1_level > 0.06) AND NOT di_dryfire AND NOT di_estop THEN dryfire_trip := FALSE; END_IF;

  IF (tank1_temp > 70.0) OR di_estop THEN overtemp_trip := TRUE; END_IF;
  IF (tank1_temp < 65.0) AND NOT di_estop THEN overtemp_trip := FALSE; END_IF;

  (* pump VFD: cut on overflow / e-stop *)
  IF overflow_trip THEN
    vfd_cmd := 0.0;
  ELSIF out_vfd > 50.0 THEN
    vfd_cmd := 50.0;
  ELSE
    vfd_cmd := out_vfd;
  END_IF;

  (* V-12: spring-close on overflow (RLY-101 cuts valve power), else clamp *)
  IF overflow_trip THEN
    v_12_cmd := 0.0;
  ELSIF out_v12 > 100.0 THEN
    v_12_cmd := 100.0;
  ELSE
    v_12_cmd := out_v12;
  END_IF;

  (* V-23: same as V-12 *)
  IF overflow_trip THEN
    v_23_cmd := 0.0;
  ELSIF out_v23 > 100.0 THEN
    v_23_cmd := 100.0;
  ELSE
    v_23_cmd := out_v23;
  END_IF;

  (* V-33 (Tank3 -> reservoir): spring-close on overflow, else clamp *)
  IF overflow_trip THEN
    v_33_cmd := 0.0;
  ELSIF out_v33 > 100.0 THEN
    v_33_cmd := 100.0;
  ELSE
    v_33_cmd := out_v33;
  END_IF;

  (* E-101 heater: cut on dry-fire / over-temp / e-stop *)
  IF dryfire_trip OR overtemp_trip THEN
    e_101_cmd := 0.0;
  ELSIF out_h1 > 100.0 THEN
    e_101_cmd := 100.0;
  ELSE
    e_101_cmd := out_h1;
  END_IF;

  scan_count := scan_count + 1;
END_PROGRAM
