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
(an under-actuated temperature problem: 4 actuators for 3 levels + 3 temps).

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
> from the approved electrical BOM ([Electrical_BOM_for_heated_tanks.md](Electrical_BOM_for_heated_tanks.md)).
> Tank geometry (37.24 L, 0.0784 m²), pump flow (4 m³/h, quadratic pump curve),
> heat-loss (4.0 W/K estimate), gravity_drop (0.3 m), and safety trips are set.
> Valve Cv is an interim estimate pending datasheets. Physics equations are
> aligned with the AIO-Gym v0.2 three_tank model (orifice valve flow, pump
> curve, hydraulic overflow, 8-dim state incl. a finite 150 L reservoir). The
> final deployment runs on a **Mac mini** connected to the real I/O via the
> RTU-to-TCP gateway.

### Process & register map

Heated serial cascade with recirculation (topology above). The rig is
**multi-slave Modbus** behind an RTU-to-TCP gateway at `127.0.0.1:5020` — five
slaves with segregated function codes:

| Slave | Module | FC | Type | Channels |
| --- | --- | --- | --- | --- |
| 02 | AI | FC04 input reg | f32 | LT-101/201/301 level (0–0.5 m), TT-101/201 temp (0–100 °C), FT-101/201/301 flow (0–50 L/min) |
| 05 | AI #2 | FC04 input reg | f32 | TT-301 temp (slave 02 full at 8 ch) |
| 03 | AO + reset | FC06 holding | u16 | V-12/V-23/E-101/V-33 cmd (0–10000 = 0–100 %), reset_cmd, init_h1–3 |
| 04 | DI | FC02 discrete | bool | dry-fire, overflow, heater/pump contactor, e-stop |
| 06 | VFD | FC06 holding | u16 | vfd_cmd — Inovance MD200 freq ref (addr 0x1000, 0–10000 = 0–100 % of F0-10) |

Analog sensors are 32-bit floats (2 registers, big-endian ABCD); actuator commands
are uint16 raw 0–10000 (FC06 single-register write, MD200-style); safety statuses
are FC02 discretes. The single source of truth is [`ia2_config.json`](ia2_config.json).

### Safety model (5 layers)

| Layer | What | Where |
| --- | --- | --- |
| L1–L4 | Hardware (RCD, high/low-level floats, capillary thermostat, contactors) | Physical plant — emulated as the 5 FC02 DI flags |
| **L5** | **Software shield** — clamps + interlocks every actuator | **`threetank.st` (this repo)** |

The L5 shield runs in the PLC scan loop. The supervisor writes `*_cmd_req` (REAL
0–100 %); the PLC converts to uint16 raw (×100 → 0–10000), clamps, and forces the
mapped output to 0 based on the 5 hardware DI flags + software level/temp limits:

- **Pump/VFD OFF** on overflow or e-stop (DI flags)
- **Valves V-12/V-23/V-33 spring-close** on overflow (RLY-101 cuts valve power)
- **Heater E-101 OFF** on dry-fire, over-temp (>70 °C), or e-stop

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
```

### Control backends

For any controller supervisor (`run_mpc.py`, `run_nmpc.py`, `run_rl.py`, `validate_policy.py`, `manual_gui.py`), you can specify the communication backend using the `--backend` flag. This allows you to bypass the IA2 server for quick tests, or target a remote edge device.

- ``auto`` (default for scripts): Uses ia2 if the dev server is running and a program is loaded, otherwise falls back to modbus.
- ``ia2``: Connects via the local IA2 dev server and runs the full PLC scan + L5 safety shield.
- ``modbus``: Bypasses IA2 and connects directly to mock_cabinet.py (useful for quick standalone tests without booting IA2).
- ``edge:<name>``: Connects to a remote edge runtime via the dev server's SSH proxy.

>Note on edge latency: If using `--backend edge:<name>`, be aware that each step requires an SSH round-trip proxied >through the dev server (~6 handshakes per step). For edge deployments, increase the step time using ``--control-dt`` (e.g., >``--control-dt 2.0``) to accommodate the network latency.

```bash
# Example: Run MPC directly on the mock cabinet (no IA2 server required)
python3 controllers/run_mpc.py --backend modbus

# Example: Run the Manual GUI against a remote edge device
python3 controllers/manual_gui.py --backend edge:my_edge --control-dt 2.0

# Example: Validate a policy on the IA2 track, explicitly enforcing the backend
python3 controllers/validate_policy.py --policy controllers/policies/sac_threetank.zip --backend ia2
```

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

| Mode | Controller | Runs in | Agent writes |
| --- | --- | --- | --- |
| Manual | `FB_MANSTATION` | PLC | `manual_*` (0–100 %) |
| PID | `FB_PID` × 4 | PLC | `*_sp` setpoints (3 levels + Tank1 temp) |
| MPC | `MPCAgent` (numpy) | Python supervisor | `*_cmd_req` |
| NMPC | `NMPCOracle` (CasADi) | Python supervisor | `*_cmd_req` |
| RL | Trained SAC/PPO | Python supervisor | `*_cmd_req` (actuator) or `*_sp` (setpoint) |

### RL training & benchmark (AIO-Gym integration)

The plant is registered as a `threetank` scenario in AIO-Gym via runtime registry
injection (`controllers/aiogym_register.py`) — so AIO-Gym's env, trainers, scorer,
and `evaluate()` all work unchanged against our plant.

**Train:**
```bash
python3 controllers/train_sb3.py --algo sac --reward-mode kpi --steps 500000 --n-envs 8
```

**Benchmark (compare all controllers on the same KPI):**
```bash
python3 controllers/benchmark.py --rl controllers/policies/sac_threetank.zip --reward-mode kpi
# add --nmpc for the CasADi NMPC oracle (slow)
```

Output (AIO-Gym-style KPI table, ranked):
```
=== Benchmark (mode=kpi, 5 eps x 200 steps) ===
controller     kpi   ±std temp_err  lvl_cm excess_kwh interlock
---------------------------------------------------------------
RL-SAC       97.09   0.63     0.51    1.94      0.019      0.00
MPC          86.33   1.32     3.41    2.79      0.256      0.00
PID          83.35   1.57     4.20    2.33      0.355      0.00
Manual       54.93   2.49     9.65    9.24      0.434      0.18
```

> NMPC is excluded by default (CasADi + IPOPT is ~1–4 s/step, adding ~20 min to
> the benchmark). Include it with `--nmpc`:
> ```bash
> python3 controllers/benchmark.py --rl controllers/policies/sac_threetank.zip --nmpc --reward-mode kpi
> ```

**Validate (sim-to-real gate — trained policy on the real IA2 track):**
```bash
./run_mode.sh rl     # runs the default SAC policy through the 50 ms scan + L5 shield
# Or specify attributes:
./run_mode.sh rl --algo ppo
./run_mode.sh rl --train_track modbus --algo sac
```

Each run produces a **KPI report + CSV + matplotlib plot** in `controllers/runs/`.

### Training & validation workflow

The end-to-end flow — train on the fast numpy plant, benchmark against
classical controllers, then validate the winner on the real IA2 track:

```
  1. TRAIN                          2. BENCHMARK                      3. VALIDATE
  ─────────                         ────────────                      ──────────
  train_sb3.py                      benchmark.py                      run_mode.sh rl
  SAC on numpy plant                PID / MPC / RL ranked             policy on IA2 track
  (in-process, fast)                by KPI score                      (50 ms scan + L5 shield)
       │                                  │                                  │
       ▼                                  ▼                                  ▼
  sac_threetank.zip               KPI table + CSV + PNG              sim-to-real gap
  + .json (action mode)           (controllers/runs/)                (numpy KPI vs IA2 KPI)
```

```bash
# step 1 — train SAC on the numpy plant (fast; ~5 min for 500k steps on GPU)
python3 controllers/train_sb3.py --algo sac --reward-mode kpi --steps 500000 --n-envs 8

# step 2 — benchmark: PID vs MPC vs RL on the same KPI yardstick
python3 controllers/benchmark.py --rl controllers/policies/sac_threetank.zip --reward-mode kpi

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

After training on the Modbus track, the **benchmark + validation steps are
identical** to the primary workflow above — just point at the Modbus-trained
policy using the `./run_mode.sh rl` attributes:

```bash
# benchmark (same KPI scorer, same plant comparison)
python3 controllers/benchmark.py --rl controllers/policies/sac_cascade.zip --reward-mode kpi

# validate (runs policy directly on mock_cabinet via Modbus backend)
./run_mode.sh rl --train_track modbus --algo sac
# or for PPO trained on Modbus:
./run_mode.sh rl --train_track modbus --algo ppo
```

~37× real-time with 4 envs × 10× time-scale (random throughput check). The IA2
validation track stays single-instance (one PROGRAM per server).

### Repository layout

```
cascade-control-sandbox/
├── ia2_config.json            # single contract — multi-slave register map, scales, setpoints
├── Electrical_BOM_for_heated_tanks.md  # source hardware spec (approved electrical BOM)
├── mock_cabinet.py            # multi-slave pymodbus TCP plant on :5020 (--time-scale k)
├── aio_bridge_env.py          # Gymnasium env (ia2 / edge / modbus backends; --mode)
├── aio_vec_env.py             # vectorized training env (N cabinets, AsyncVectorEnv)
├── run_mode.sh                # boot + run one controller + teardown (one command)
├── tools/
│   └── gen_ia2_artifacts.py   # contract → device/iomap TOMLs (+ ST VAR check)
├── controllers/
│   ├── threetank_model.py     # numpy heated serial-cascade plant (AIO-Gym model interface)
│   ├── mpc_agent.py           # numpy MPC (box-QP)
│   ├── nmpc_oracle.py         # CasADi+IPOPT NMPC (symbolic plant)
│   ├── run_mpc.py             # MPC supervisor (IA2 track)
│   ├── run_nmpc.py            # NMPC supervisor (IA2 track)
│   ├── run_rl.py              # RL supervisor (trained policy on IA2 track)
│   ├── train_sb3.py           # SAC/PPO training (AIO-Gym env + SB3)
│   ├── train_rl.py            # vectorized training (Modbus track)
│   ├── benchmark.py           # KPI benchmark (PID/MPC/NMPC/RL ranked)
│   ├── aiogym_register.py     # register "threetank" in AIO-Gym's registries
│   ├── validate_policy.py     # sim-to-real validation gate
│   ├── rollout_report.py      # shared KPI table + CSV + PNG plot
│   ├── manual_gui.py          # tkinter manual control GUI
│   └── policies/              # trained RL policies (.zip) + action-mode .json sidecars (.zip gitignored)
├── tests/
│   ├── smoke_reset.py         # reset snaps levels to targets
│   ├── smoke_heater.py        # heater raises temp; cold pump inflow slows it
│   ├── smoke_env.py           # env reset/step/reward over Modbus
│   └── run_smoke.sh           # one-command runner (boots + tests + teardown)
├── ia2_project/               # IA2 PLC project (IEC 61131-3 ST + device + iomap)
│   ├── devices/cabinet_*.toml      # AUTO-GENERATED — 5 Modbus devices (one per slave)
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

| OS | IA2 build | CUDA (for RL training) | Manual GUI |
| --- | --- | --- | --- |
| **WSL2 (Windows)** | `cargo build --release` in WSL | `pip install torch` (auto-detects CUDA via WSL GPU passthrough) | `sudo apt install python3-tk` (renders via WSLg on Windows 11) |
| **Linux (native)** | `cargo build --release` | `pip install torch` (CUDA if NVIDIA GPU present, else CPU) | `sudo apt install python3-tk` or `python3-tkinter` |
| **macOS** | `cargo build --release` | `pip install torch` (CPU — MPS per-op overhead dominates small MLPs; use `--device cpu`) | Bundled with system Python (no install needed) |

> The `best_device()` helper in `train_sb3.py` auto-selects CUDA → CPU. Override
> with `--device mps` if you want to try Apple Silicon.

### Quick start

```bash
# try PID control (boots everything, runs, tears down)
./run_mode.sh pid

# benchmark all controllers
python3 controllers/benchmark.py --rl controllers/policies/sac_threetank.zip

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
./tests/run_smoke.sh     # boots the cabinet, runs all three, tears down
```

- `smoke_reset.py` — reset snaps tank levels to requested targets
- `smoke_heater.py` — heater raises temp; cold pump inflow slows it (the cascade)
- `smoke_env.py` — env resets (randomized), steps, and rewards over Modbus

### Deployment (sim → hardware)

IA2 fronts the plant through the device config, not in code. Moving from the local
simulator to physical hardware is a configuration change: repoint the
`ia2_project/devices/cabinet_*.toml` device files at the real RTU-to-TCP gateway
IP + slave addresses (02/03/04/05/06), and align the channel addresses/scales to
the real I/O map. The PLC program, iomap, and bridge env run unchanged against
real hardware.
