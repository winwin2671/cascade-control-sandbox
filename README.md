<div align="center">

# cascade-control-sandbox

</div>

---

A reinforcement-learning sandbox for **control of a heated serial-cascade +
recirculation 3-tank process**, built on the [IA2](https://github.com/supcon-international/ia2)
industrial-automation engine and the [AIO-Gym](https://github.com/supcon-international/AIO-Gym)
agent-in-the-loop pattern. The physical "cabinet" is a Python **multi-slave Modbus
TCP simulator** (32-bit float analog channels + FC02 discretes, behind an
RTU-to-TCP gateway); IA2 fronts it as a real PLC would front real hardware; a
Gymnasium environment bridges an RL agent to IA2.

**Plant topology:** pump P-101 → Tank1 →(valve V-12)→ Tank2 →(valve V-23)→
Tank3 →(valve V-33)→ Reservoir →(P-101 on VFD)→ Tank1 (recirculation). One 2 kW
heater (E-101) in Tank1 only — Tank2/Tank3 warm via downstream hot-water advection
(an under-actuated temperature problem: 5 actuators for 3 levels + 3 temps).

### Highlights

- **Five control modes, one plant.** Manual (tkinter GUI), PID (PLC `FB_PID`),
  MPC (numpy box-QP), NMPC (CasADi + IPOPT), and RL (trained SAC/PPO) — switch
  live, compare on the same KPI.
- **Real PLC, not a toy.** The `threetank.st` IEC 61131-3 program runs in IA2's
  50 ms scan loop with a 5-layer safety architecture (L5 software shield reacts
  to 5 FC02 hardware DI flags — overflow, dry-fire, over-temp, contactors, e-stop).
- **AIO-Gym integration for MPC & RL.** IA2 ships native PID + Manual (IEC 61131-3
  FBs), but does not yet include MPC or RL controllers. This sandbox imports
  AIO-Gym's implementations (numpy MPC, CasADi NMPC, SAC/PPO/RLPD training)
  rather than rewriting them — registered as a `threetank` scenario via runtime
  injection, no AIO-Gym source modification.
- **Benchmark with KPI reports.** Run PID / MPC / NMPC / RL head-to-head, ranked
  by the composite KPI score (tracking + energy + safety). Each run produces a
  KPI table + CSV + matplotlib plot.
- **Fast training track.** `--time-scale 10` accelerates the physics 10×;
  `AsyncVectorEnv` runs N cabinets in parallel — ~37× real-time with 4 envs.
- **Sim-to-real validation gate.** Load a trained policy, run it through the IA2
  track (real scan + iomap + L5 shield), compare the KPI to the numpy benchmark.
- **Zero-drag contract.** `ia2_config.json` is the single source of truth — a
  code generator emits the device/iomap TOMLs + validates the ST declarations.

### Architecture

```
      +-------------------------------------------+
      |        RL Agent / AIO-Gym (Strategy)      |
      +-------------------------------------------+
                            |
                     (Gymnasium API)
                            |
      +-------------------------------------------+
      |         aio_bridge_env.py (Bridge)        |
      +-------------------------------------------+
                            |
                     (HTTP /api/runtime)
                            |
      +-------------------------------------------+
      |     IA2 Automation Engine (Execution)     |
      +-------------------------------------------+
                            |
               (Modbus TCP 127.0.0.1:5020)
                            |
      +-------------------------------------------+
      |       mock_cabinet.py (Process Plant)     |
      +-------------------------------------------+
```

> **Why AIO-Gym?** IA2 does not yet ship its own MPC or RL controllers. For PID
> and Manual it uses its native IEC 61131-3 `FB_PID` / `FB_MANSTATION` function
> blocks. For MPC (numpy + CasADi NMPC) and RL (SAC / PPO / RLPD), the sandbox
> imports AIO-Gym's battle-tested implementations rather than rewriting them. When
> IA2 gains native MPC/RL, the AIO-Gym dependency can be dropped.

> **Simulation status:** `mock_cabinet.py` models the heated serial-cascade rig
> from the approved electrical BOM (the BOM document itself is kept offline —
> not committed to the repo).
> Tank geometry (37.24 L, 0.0784 m²), pump flow (4 m³/h, quadratic pump curve),
> heat-loss (4.0 W/K estimate), gravity_drop (0.3 m), and safety trips are set.
> Valve Cv is an interim estimate pending datasheets. Physics equations are
> aligned with the AIO-Gym v0.2 three_tank model (orifice valve flow, pump
> curve, hydraulic overflow, 8-dim state incl. a finite 150 L reservoir). The
> final deployment runs on a **Mac mini** connected to the real I/O via the
> RTU-to-TCP gateway.

### Process & register map

Heated serial cascade with recirculation (topology above). The rig is
**multi-slave Modbus** behind an RTU-to-TCP gateway at `127.0.0.1:5020` — seven
slaves with segregated function codes:

| Slave | Module   | FC             | Type | Channels                                                                                     |
| ----- | -------- | -------------- | ---- | -------------------------------------------------------------------------------------------- |
| 02    | AI       | FC04 input reg | f32  | LT-101/201/301 level (0–0.5 m), TT-101/201 temp (0–100 °C), FT-101/201/301 flow (0–50 L/min) |
| 05    | AI #2    | FC04 input reg | f32  | TT-301 temp (slave 02 full at 8 ch)                                                          |
| 03    | AO       | FC06 holding   | u16  | V-12/V-23/E-101/V-33 cmd (0–10000 = 0–100 %)                                                 |
| 01    | sim-only | FC06 holding   | u16  | reset_cmd, init_h1–3 (episode reset; a unit id the real gateway doesn't occupy)              |
| 04    | DI       | FC02 discrete  | bool | dry-fire, overflow, heater/pump contactor, e-stop                                            |
| 06    | VFD      | FC06 holding   | u16  | vfd_cmd — Inovance MD200 freq ref (addr 0x1000, 0–10000 = 0–100 % of F0-10)                  |
| 07    | DO       | FC05 coil      | bool | SV-1/2/3 on/off interlock-test solenoids (each parallel to V-12/V-23/V-33)                   |

SV-1..3 are **test instrumentation, not control actuators**: they are excluded
from the RL action space (config `test_actuators` vs `actuators`) and driven
only through `env.set_test_valve()` / `env.set_test_valves_enabled()` — on the
IA2 track the POU ANDs each request with `test_sv_en` (default FALSE) before
energizing the coil, so stale requests can't hold a test valve open. Opening one
gives full-bore flow on that path regardless of valve position — the scripted
level transient for SAT interlock testing (drive a tank to the LSH overflow or
LSL dry-fire trip and verify the response). FT-10x read the combined
valve+bypass line, so an open SV is directly observable on the flow sensors.

Analog sensors are 32-bit floats (2 registers, big-endian ABCD); actuator commands
are uint16 raw 0–10000 (FC06 single-register write, MD200-style); safety statuses
are FC02 discretes. The single source of truth is [`ia2_config.json`](ia2_config.json).

### Safety model (5 layers)

| Layer  | What                                                                    | Where                                            |
| ------ | ----------------------------------------------------------------------- | ------------------------------------------------ |
| L1–L4  | Hardware (RCD, high/low-level floats, capillary thermostat, contactors) | Physical plant — emulated as the 5 FC02 DI flags |
| **L5** | **Software shield** — clamps + interlocks every actuator                | **`threetank.st` (this repo)**                   |

The L5 shield runs in the PLC scan loop. The supervisor writes `*_cmd_req` (REAL
0–100 %); the PLC converts to uint16 raw (×100 → 0–10000), clamps, and applies
the interlock latches (5 hardware DI flags + software level/temp limits):

- **Pump/VFD OFF** on overflow or e-stop (DI flags) — overflow stops _inflow_ only
- **Drains stay available in a latch** — valves are never cut on overflow (cutting
  them sustained the overflow and made the latch unrecoverable); while a tank is
  above the 0.40 recovery threshold its drain is forced open so the level can fall
- **Heater E-101 OFF** on dry-fire, over-temp (>70 °C), or e-stop

> Design rule: a trip may only inhibit the actuator that _aggravates_ it. Note the
> BOM's RLY-101 relay still cuts valve power on overflow **in hardware** — the
> software shield deliberately doesn't; recommend repurposing that relay to the
> pump contactor before commissioning so both layers agree.

### Control modes

`run_mode.sh` boots the cabinet + IA2 + the PLC, runs the chosen controller, and
tears down on exit — one command per mode:

```bash
./run_mode.sh pid       # PLC FB_PID tracks the config setpoints
./run_mode.sh manual    # operator manual_* → FB_MANSTATION → actuators
./run_mode.sh mpc       # numpy MPC (successive-linearization box-QP)
./run_mode.sh nmpc      # CasADi + IPOPT NMPC (offline oracle; ~1–4 s/step)
./run_mode.sh rl [opts] # trained SAC/PPO policy (setpoint supervisory mode)
./run_mode.sh gui       # interactive tkinter GUI (sliders + live plot + KPI)
./run_mode.sh modbus    # direct Modbus (skip IA2; for quick standalone tests)
./run_mode.sh pid --steps 40  # more steps (pid/manual/rl/modbus; mpc/nmpc run 40)
./run_mode.sh pid --disturbance  # any mode + random SV-1..3 valve-fault disturbances
```

### Control backends

For any controller supervisor (`run_mpc.py`, `run_nmpc.py`, `run_rl.py`, `validate_policy.py`, `manual_gui.py`), you can specify the communication backend using the `--backend` flag. This allows you to bypass the IA2 server for quick tests, or target a remote edge device.

- `auto` (default for scripts): Uses ia2 if the dev server is running and a program is loaded, otherwise falls back to modbus.
- `ia2`: Connects via the local IA2 dev server and runs the full PLC scan + L5 safety shield.
- `modbus`: Bypasses IA2 and connects directly to mock_cabinet.py (useful for quick standalone tests without booting IA2).
- `edge:<name>`: Connects to a remote edge runtime via the dev server's SSH proxy.

> Note on edge latency: If using `--backend edge:<name>`, be aware that each step requires an SSH round-trip proxied >through the dev server (~6 handshakes per step). For edge deployments, increase the step time using `--control-dt` (e.g., >`--control-dt 2.0`) to accommodate the network latency.

```bash
# Example: Run MPC directly on the mock cabinet (no IA2 server required)
python3 controllers/run_mpc.py --backend modbus

# Example: Run the Manual GUI against a remote edge device
python3 controllers/manual_gui.py --backend edge:my_edge --control-dt 2.0

# Example: Validate a policy on the IA2 track, explicitly enforcing the backend
python3 controllers/validate_policy.py --policy controllers/policies/sac_threetank_numpy.zip --backend ia2
```

### Disturbance / interlock testing (`--disturbance`)

Add `--disturbance` to any mode to run [`disturbance_sidecar.py`](disturbance_sidecar.py) in the
background alongside the controller. The schedule mirrors AIO-Gym's auto-events (`_autoTick`):
a single event clock fires every 10–32 s; each event closes every SV then either opens **one**
valve (uniform among the chosen set — the fault persists until the *next* event clears it) or
declares a **quiet period** (~30% of events) so the plant gets recovery time. One fault at a
time, never overlapping — like real process control, where an equipment fault latches until
cleared. Levels drift on the bypass path, FT-10x see the extra flow, and LSH/LSL trips become
reachable. At the default cadence a 60-step (30 s) run sees only ~1 event — for controller
comparisons either lengthen the run or compress the clock, e.g.
`DISTURBANCE_ARGS="--event-min 4 --event-max 10" ./run_mode.sh mpc --steps 200 --disturbance`.

- **Write path follows the track** (derived from the mode): on IA2 modes the PLC owns the SV
  coils, so the sidecar writes the PLC-internal `sv_*_req` vars + `test_sv_en` gate through the
  IA2 variable API; only the PLC-less paths (`modbus`, `rl --train_track modbus`) write the
  slave-07 FC05 coils directly. Never write the coils directly while the PLC runs — its scan
  reclaims them.
- **Reproducible**: the seed is printed and logged; replay a run with `--seed N` (plus the same
  `--valves`, `--event-min/max`, and `--quiet-prob`, all captured in the JSONL header). Tune the
  sidecar via `DISTURBANCE_ARGS` (e.g. `DISTURBANCE_ARGS="--seed 42 --event-min 4" ./run_mode.sh pid --disturbance`)
  or run it standalone against real hardware: `python3 disturbance_sidecar.py --backend modbus
  --host <gateway>` (or `--backend ia2 --server <url>`).
- **Survives resets**: the intended valve state is re-asserted every 0.5 s, so `env.reset()`
  force-closing the SVs at episode start can't swallow a disturbance — the hold resumes within
  one heartbeat (episodic RL eval sees faults continue mid-hold).
- **Cleanup**: on SIGINT/SIGTERM the sidecar closes every SV and clears `test_sv_en` before
  `run_mode.sh` tears down the cabinet. `kill -9` skips that — coils keep their last state until
  the next boot. Concurrent sidecars are unsupported (two schedules fight over the coils).
- **Log**: every run writes `controllers/runs/disturbance_sidecar_YYYYMMDD_HHMMSS.jsonl`
  (header with seed/params, one record per transition, exit footer) — the record of what a
  rollout saw; correlate it offline with the rollout CSVs.
- **Reports**: during `--disturbance` runs the controller's rollout artifacts are tagged
  `_dist` (`pid_dist_rollout.csv/png`, `rl_dist_rollout.csv`, …) so they never overwrite the
  baseline `<tag>_rollout` files — diff the pair to see what the faults cost the controller.

For the **RL mode** (`./run_mode.sh rl`), you can specify the following attributes to match how the policy was trained:

- `--algo <sac|ppo>`: Specify the algorithm (defaults to `sac`).
- `--train_track <numpy|modbus>`: Specify the training track (defaults to `numpy`). If `modbus` is used, the script automatically skips the IA2 server and connects directly to the `mock_cabinet.py` plant, matching how cascade policies were trained.

```bash
# Examples for RL mode attributes:
./run_mode.sh rl                                  # default: --algo sac --train_track numpy
./run_mode.sh rl --algo ppo                       # evaluate a PPO policy trained on numpy
./run_mode.sh rl --train_track modbus             # evaluate a SAC policy trained on Modbus
./run_mode.sh rl --algo ppo --train_track modbus  # evaluate a PPO policy trained on Modbus
```

| Mode   | Controller            | Runs in           | Agent writes                                |
| ------ | --------------------- | ----------------- | ------------------------------------------- |
| Manual | `FB_MANSTATION`       | PLC               | `manual_*` (0–100 %)                        |
| PID    | `FB_PID` × 4          | PLC               | `*_sp` setpoints (3 levels + Tank1 temp)    |
| MPC    | `MPCAgent` (numpy)    | Python supervisor | `*_cmd_req`                                 |
| NMPC   | `NMPCOracle` (CasADi) | Python supervisor | `*_cmd_req`                                 |
| RL     | Trained SAC/PPO       | Python supervisor | `*_cmd_req` (actuator) or `*_sp` (setpoint) |

### RL training & benchmark

The plant supports **two RL paradigms** and **four reward modes**, aligned with the
AIO-Gym v0.2 model (xinji's three_tank). Physics equations are shared across the
mock, MPC model, and NMPC oracle via [`controllers/threetank_dynamics.py`](controllers/threetank_dynamics.py).

#### Two RL paradigms

The single flag `--residual` selects the paradigm on **either** track:

| Paradigm                     | `--residual` | Action dim | What the RL controls                                          | Reward                              |
| ---------------------------- | ------------ | ---------- | ------------------------------------------------------------- | ----------------------------------- |
| **Direct actuator**          | _off_ (omit) | **5D**     | all MVs: pump + V-12 + V-23 + V-33 + heater                   | configurable (`--reward-mode`)      |
| **Residual** (xinji-aligned) | **on**       | **2D**     | only V-33 + heater (on top of model feedforward + inline PID) | regulation (tracking + feedforward) |

Both paradigms are wired into **both** trainers — `train_sb3.py` (numpy) and
`train_rl.py` (modbus) — so you can compare 5D vs 2D on the **same** environment
and isolate the paradigm (not the track):

```bash
# numpy track: SAME env, SAME step budget, SAME seeds → fair 5D-vs-2D bake-off
python3 controllers/train_sb3.py --algo sac --action-mode actuator --reward-mode regulation --steps 500000   # → sac_threetank_numpy.zip  (5D)
python3 controllers/train_sb3.py --algo sac --residual --reward-mode regulation --steps 500000               # → sac_residual_numpy.zip  (2D)
```

The **residual paradigm** (from xinji's v0.2) trains faster and is more stable:
the RL agent only adjusts 2 actuators (V-33 drain + heater), while a model-based
feedforward controller (`tracking_steady_state_action`) + inline PID handle the
upstream hydraulics (pump, V-12, V-23). The reward penalizes deviation from the
model-predicted action (feedforward term), keeping the RL near the physics-optimal
baseline unless it can improve tracking.

> **Fair comparison rule:** never compare a numpy-trained policy against a
> modbus-trained policy directly — the tracks differ in speed, step budget, and
> observation/latency, which confounds the result. Hold the track constant
> (pick one) and vary only `--residual` on/off.

#### Reward modes

| Mode                             | Formula                                                 | Tracks                          |
| -------------------------------- | ------------------------------------------------------- | ------------------------------- |
| `economic` (v0.1 default)        | `-(w_energy × heater_power + w_viol × band_violations)` | numpy                           |
| `kpi`                            | composite KPI score (tracking + energy + safety)        | numpy                           |
| `track`                          | `-(level_MSE + temp_MSE + action_cost)`                 | numpy + modbus                  |
| **`regulation`** (xinji-aligned) | `-(tracking + 0.03 × feedforward_deviation)`            | **numpy + modbus** (drift-free) |

The `regulation` reward is the **recommended mode** — it's aligned with xinji's
v0.2 model and produces the same reward on both tracks (eliminating the reward
drift between numpy and modbus training).

**Train (modbus track — sim-to-real, slower):**

```bash
# 2D residual (xinji-aligned: V-33 + heater on top of model feedforward)
python3 controllers/train_rl.py --residual --algo sac --total-timesteps 50000 --device cuda
# 5D direct actuator (EnrichedObs)
python3 controllers/train_rl.py --algo sac --total-timesteps 50000
```

**Benchmark (compare all controllers on the same KPI):**

```bash
python3 controllers/benchmark.py --rl controllers/policies/sac_residual_numpy.zip --reward-mode kpi --episode-steps 4000
# add --nmpc for the CasADi NMPC oracle (slow)
```

Current result (residual SAC, 500k steps × 4000-step episodes,
fixed-reference eval — 20 eps × 4000 steps, kpi mode):

```
controller     kpi   ±std temp_err  lvl_cm excess_kwh interlock
---------------------------------------------------------------
RL-SAC-res   71.62   1.23    13.95    0.30     0.267      0.00
MPC          69.75   2.98    14.67    0.98     0.137      0.00
PID          66.52   0.89    15.21    3.17     0.581      0.00
Manual       43.06   1.40    17.89   26.46     0.000      0.00
```

**Validate (sim-to-real gate — trained policy on the live track):**

Any of the four train combinations validates the same way — `--train_track` selects
numpy (IA2 backend) or modbus, and `--residual` selects the 2D policy (off = 5D):

| Train combination    | Validate command                                   |
| -------------------- | -------------------------------------------------- |
| numpy + 5D direct    | `./run_mode.sh rl`                                 |
| numpy + 2D residual  | `./run_mode.sh rl --residual`                      |
| modbus + 5D direct   | `./run_mode.sh rl --train_track modbus`            |
| modbus + 2D residual | `./run_mode.sh rl --train_track modbus --residual` |

```bash
./run_mode.sh rl                       # default: numpy-track, 5D, algo=sac → sac_threetank_numpy.zip
./run_mode.sh rl --residual            # numpy-track, 2D residual            → sac_residual_numpy.zip
./run_mode.sh rl --train_track modbus  # modbus-track, 5D                    → sac_threetank_modbus.zip
./run_mode.sh rl --algo ppo            # swap algorithm
```

`run_mode.sh` resolves the policy file (`${algo}_threetank_<track>` / `_residual_<track>`)
and the action mode (5D `actuator` / 2D `residual`) automatically; `run_rl.py` wraps
the eval env with `ResidualEnvWrapper` for a 2D policy so its action expands to the
full 5D physical action before stepping the plant. The wrapper's 23-D observation (17 +
6 integral-of-error terms — the I-term for warm-up persistence and offset-free holding) is
backend-agnostic, so a numpy-trained residual policy validates on either backend.

Each run produces a **KPI report + CSV + matplotlib plot** in `controllers/runs/`.

### Training & validation workflow

```
  1. TRAIN                          2. BENCHMARK                      3. VALIDATE
  ─────────                         ────────────                      ──────────
  train_sb3.py (numpy)              benchmark.py                      run_mode.sh rl
  or train_rl.py (modbus)           PID / MPC / RL ranked             policy on IA2 track
                                     by KPI score                      (50 ms scan + L5 shield)
       │                                  │                                  │
       ▼                                  ▼                                  ▼
  sac_threetank_numpy.zip         KPI table + CSV + PNG              sim-to-real gap
  or sac_residual_numpy.zip       (controllers/runs/)                (numpy KPI vs IA2 KPI)
```

```bash
# step 1 — train (numpy track, ~5 min for 500k steps on GPU)
python3 controllers/train_sb3.py --algo sac --reward-mode regulation \
    --action-mode actuator --steps 500000 --n-envs 8

# step 2 — benchmark: PID vs MPC vs RL on the same KPI yardstick
python3 controllers/benchmark.py --rl controllers/policies/sac_threetank_numpy.zip --reward-mode kpi

# step 3 — validate: run the trained policy on the real IA2 track (boots everything)
./run_mode.sh rl                                  # defaults to --algo sac --train_track numpy
./run_mode.sh rl --algo ppo                       # validate a numpy-trained PPO policy
./run_mode.sh rl --train_track modbus --algo sac  # validate a Modbus-trained SAC policy
```

### Manual control GUI

```bash
./run_mode.sh gui
```

Launches a tkinter desktop window (rendered on Windows via WSLg):

- **5 sliders** (Pump/VFD, V-12, V-23, V-33, Heater/E-101 — 0–100 %)
- **DI safety readout** — 5 FC02 hardware flags (dry-fire, overflow, contactors, e-stop)
- **Reset button** — re-pulses the reset nonce; tanks snap back to the startup init levels and the episode KPI/history resets
- **Real-time plot** — levels + temps with setpoint lines
- **Live KPI readout** — score, temp error, level error

### Training track (Modbus, secondary path)

> **`train_rl.py` (Modbus) vs `train_sb3.py` (numpy):** `train_sb3.py` is the
> **primary training path** — it trains on the numpy `ThreeTankModel`
> (in-process, thousands of steps/s). `train_rl.py` is a **secondary path** —
> it trains on the actual `mock_cabinet.py` via Modbus TCP (slower, but
> verifies the mock_cabinet physics match the numpy model, and tests training
> directly against the simulated hardware). Most users should use `train_sb3.py`
> for training; `train_rl.py` for physics-drift testing.

```bash
# random policy — throughput check (gymnasium AsyncVectorEnv)
python3 controllers/train_rl.py --n-envs 4 --time-scale 10 --steps 200

# PPO training on the Modbus track (SB3 DummyVecEnv over N cabinets)
python3 controllers/train_rl.py --algo ppo --n-envs 4 --time-scale 10 \
    --total-timesteps 50000 --device cpu

# SAC training on the Modbus track
python3 controllers/train_rl.py --algo sac --n-envs 4 --time-scale 10 \
    --total-timesteps 50000 --device cuda

# one cabinet k× faster (for standalone testing)
python3 mock_cabinet.py --time-scale 10
```

> **PPO on CPU:** small MLP policies (256×256) train faster on CPU than GPU
> (per-op overhead dominates tiny matmuls). Pass `--device cpu` for PPO; SAC
> benefits from CUDA (`--device cuda`).

After training on the Modbus track, **validation is identical** to the primary
workflow above — just point at the Modbus-trained policy using the `./run_mode.sh rl`
attributes:

```bash
# validate (runs policy directly on mock_cabinet via Modbus backend)
./run_mode.sh rl --train_track modbus --algo sac
# or for PPO trained on Modbus:
./run_mode.sh rl --train_track modbus --algo ppo
```

> `benchmark.py` accepts numpy-trained policies (and residual policies of either
> track) but **rejects modbus-trained direct policies** up front: their
> EnrichedObs includes bridge-only sensors (3 flows + 5 DI flags) the numpy env
> doesn't produce. Evaluate those on the bridge instead:
> `python3 controllers/run_rl.py --policy controllers/policies/sac_threetank_modbus.zip --backend ia2`

~37× real-time with 4 envs × 10× time-scale (random throughput check). The IA2
validation track stays single-instance (one PROGRAM per server).

### Repository layout

```
cascade-control-sandbox/
├── ia2_config.json            # single contract — multi-slave register map, scales, setpoints
├── mock_cabinet.py            # multi-slave pymodbus TCP plant on :5020 (--time-scale k)
├── aio_bridge_env.py          # Gymnasium env (ia2 / edge / modbus backends; --mode)
├── aio_vec_env.py             # vectorized training env (N cabinets, AsyncVectorEnv)
├── disturbance_sidecar.py     # random SV-1..3 valve-fault injection (--disturbance)
├── run_mode.sh                # boot + run one controller + teardown (one command)
├── tools/
│   └── gen_ia2_artifacts.py   # contract → device/iomap TOMLs (+ ST VAR check)
├── controllers/
│   ├── threetank_model.py     # numpy heated serial-cascade plant (AIO-Gym model interface)
│   ├── threetank_dynamics.py  # shared 8-dim ODE (numpy + CasADi ops) — one physics source
│   ├── mpc_agent.py           # numpy MPC (box-QP)
│   ├── nmpc_oracle.py         # CasADi+IPOPT NMPC (symbolic plant)
│   ├── run_mpc.py             # MPC supervisor (IA2 track)
│   ├── run_nmpc.py            # NMPC supervisor (IA2 track)
│   ├── run_rl.py              # RL supervisor (trained policy on IA2 track)
│   ├── train_sb3.py           # SAC/PPO training (AIO-Gym env + SB3)
│   ├── train_rl.py            # vectorized training (Modbus track)
│   ├── residual_rl.py         # residual RL wrappers (2D action + regulation reward + inline PID)
│   ├── benchmark.py           # KPI benchmark (PID/MPC/NMPC/RL ranked)
│   ├── aiogym_register.py     # register "threetank" in AIO-Gym's registries
│   ├── validate_policy.py     # sim-to-real validation gate
│   ├── rollout_report.py      # shared KPI table + CSV + PNG plot
│   ├── manual_gui.py          # tkinter manual control GUI
│   └── policies/              # trained RL policies (.zip) + action-mode .json sidecars (.zip gitignored)
├── tests/
│   ├── shield_regression.py   # ST safety layer under test (boots the real IA2 chain)
│   ├── smoke_reset.py         # reset snaps levels to targets
│   ├── smoke_heater.py        # heater raises temp; cold pump inflow slows it
│   ├── smoke_env.py           # env reset/step/reward over Modbus
│   ├── smoke_sv.py            # SV-1 bypass: flow + level shift on command
│   ├── smoke_disturbance.py   # disturbance sidecar follows its seeded schedule
│   └── run_smoke.sh           # one-command runner (boots + tests + teardown)
├── ia2_project/               # IA2 PLC project (IEC 61131-3 ST + device + iomap)
│   ├── devices/cabinet_*.toml      # AUTO-GENERATED — 7 Modbus devices (one per slave)
│   ├── iomap.toml                  # AUTO-GENERATED — variable ⇄ channel bindings
│   ├── tasks.toml                  # 50 ms cyclic task
│   └── pous/
│       ├── threetank.st            # mode selector (CASE) + L5 software shield
│       ├── fb_pid.st               # vendored from ia2/library/process-control
│       └── fb_manstation.st        # vendored from ia2/library/process-control
├── ia2/                       # vendored IA2 engine (gitignored; clone separately)
├── requirements.txt
└── README.md
```

### Setup

**Prerequisites:** Python ≥3.10 (pymodbus ≥3.13 floor — check with `python3 --version`).
Rust toolchain ([rustup.rs](https://rustup.rs)) for building IA2.

**Common (all OS):**

```bash
# clone this repo
git clone https://github.com/winwin2671/cascade-control-sandbox
cd cascade-control-sandbox

# Python deps (no sudo needed)
pip3 install --user -r requirements.txt

# clone AIO-Gym as a sibling (imported via sys.path — no pip install needed)
git clone https://github.com/supcon-international/AIO-Gym ../AIO-Gym

# clone + build IA2 (one-time; ~10–15 min; needs Rust toolchain — rustup.rs)
git clone --recursive https://github.com/supcon-international/ia2 ia2
cd ia2 && cargo build --release && cd ..

# regenerate the IA2 device/iomap TOMLs from the contract
python3 tools/gen_ia2_artifacts.py
```

**RL training (optional — heavy deps):**

```bash
pip3 install --user torch stable_baselines3    # see CUDA notes per-OS below
```

**OS-specific notes:**

| OS                 | IA2 build                      | CUDA (for RL training)                                                                   | Manual GUI                                                     |
| ------------------ | ------------------------------ | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| **WSL2 (Windows)** | `cargo build --release` in WSL | `pip install torch` (auto-detects CUDA via WSL GPU passthrough)                          | `sudo apt install python3-tk` (renders via WSLg on Windows 11) |
| **Linux (native)** | `cargo build --release`        | `pip install torch` (CUDA if NVIDIA GPU present, else CPU)                               | `sudo apt install python3-tk` or `python3-tkinter`             |
| **macOS**          | `cargo build --release`        | `pip install torch` (CPU — MPS per-op overhead dominates small MLPs; use `--device cpu`) | Bundled with system Python (no install needed)                 |

> The `best_device()` helper in `train_sb3.py` auto-selects CUDA → CPU. Override
> with `--device mps` if you want to try Apple Silicon.

### Quick start

```bash
# try PID control (boots everything, runs, tears down)
./run_mode.sh pid

# benchmark all controllers
python3 controllers/benchmark.py --rl controllers/policies/sac_threetank_numpy.zip

# interactive manual control GUI
./run_mode.sh gui

# run the smoke tests
./tests/run_smoke.sh
```

### Run with IA2 WebUI

To drive the plant via the **IA2 WebUI** (useful for inspecting the live PLC scan), you'll need three terminals.

**Terminal 1 — start the simulated cabinet:**

```bash
python3 mock_cabinet.py &
```

**Terminal 2 — build & launch the IA2 web server:**

```bash
cd ia2

# one-time setup
. "$HOME/.cargo/env"
pnpm install
cargo test -p server                 # populates apps/web/src/types/generated/

# build frontend + run server
pnpm --filter @cs/web build
cargo run -p server --release -- --static-dir apps/web/dist
```

Open the server URL and press **Start** on the `threetank` PRG.

**Terminal 3 — drive the plant via the bridge env:**

```bash
python3 aio_bridge_env.py --backend ia2 --mode pid --steps 200
python3 aio_bridge_env.py --backend ia2 --mode mpc --steps 200
python3 aio_bridge_env.py --backend ia2 --mode rl  --steps 200
```

> **Tip:** Use `--backend modbus` instead of `ia2` in Terminal 3 for quick standalone tests without booting the IA2 server.

### Smoke tests

```bash
./tests/run_smoke.sh     # boots the cabinet, runs all five, tears down
```

- `smoke_reset.py` — reset snaps tank levels to requested targets
- `smoke_heater.py` — heater raises temp; cold pump inflow slows it (the cascade)
- `smoke_env.py` — env resets (randomized), steps, and rewards over Modbus
- `smoke_sv.py` — SV-1 bypass: FT-101 flow + Tank1→Tank2 level shift on command, stops on close
- `smoke_disturbance.py` — seeded sidecar: coils follow the deterministic schedule (`simulate()`
  oracle), FT-101 rises while SV-1 is energized, everything closes on exit

### Shield regression test

```bash
python3 tests/shield_regression.py   # ~90 s (4x time-scale; SHIELD_TIME_SCALE=1 for real-time)
```

The smoke suite bypasses IA2, so it cannot see the PLC safety layer — the E-stop
polarity bug (#6 B1) sat exactly in that blind spot. This test closes it: it boots
the **real chain** (cabinet + ia2-server + the ThreeTank POU via `cs`) and asserts
the L5 shield through the physical path (commands → physics → sensors/DI → shield
→ mapped outputs):

- **S0 healthy passthrough** — no latch with all DIs healthy (B1 regression guard)
- **S1 overflow** — pump cut (inflow stopped), drains forced open against a shut
  command, autonomous recovery below the 0.40 threshold
- **S2 dry-fire** — heater cut, pump NOT cut (fails safe, recoverable)
- **S3 e-stop** — pressed via `cs runtime force` (NC: 0 = pressed) cuts pump +
  heater; released recovers

Falsifiability is verified by mutation: reverting the B1 polarity line makes three
scenarios fail loudly. Run it before any PR that touches `threetank.st` — it is
kept out of `run_smoke.sh` on purpose (needs the full chain, ~90 s vs seconds).

### Deployment (sim → hardware)

IA2 fronts the plant through the device config, not in code. Moving from the local
simulator to physical hardware is a configuration change: repoint the
`ia2_project/devices/cabinet_*.toml` device files at the real RTU-to-TCP gateway
IP + slave addresses (02/03/04/05/06), and align the channel addresses/scales to
the real I/O map. The PLC program, iomap, and bridge env run unchanged against
real hardware.
