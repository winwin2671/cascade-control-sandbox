<div align="center">

# cascade-control-sandbox

</div>

---

A reinforcement-learning sandbox for **cascade / decoupling control of a 3-tank
process**, built on the [IA2](https://github.com/supcon-international/ia2)
industrial-automation engine and the
[AIO-Gym](https://github.com/supcon-international/AIO-Gym) agent-in-the-loop
pattern. The physical "cabinet" is a Python Modbus TCP simulator; IA2 fronts it
as a real PLC would front real hardware; a Gymnasium environment bridges an RL
agent to IA2.

### Highlights

- **Five control modes, one plant.** Manual (tkinter GUI), PID (PLC `FB_PID`),
  MPC (numpy box-QP), NMPC (CasADi + IPOPT), and RL (trained SAC/PPO) — switch
  live, compare on the same KPI.
- **Real PLC, not a toy.** The `threetank.st` IEC 61131-3 program runs in IA2's
  50 ms scan loop with a 5-layer safety architecture (L5 software shield
  preempts hardware trips — high-level pump cutoff, dry-fire + over-temp
  heater cutoff).
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

> **Simulation status:** `mock_cabinet.py` is a **testing prototype** — the
> physics, register map, and scaling will change to match the real hardware
> design. Its purpose today is to validate the full connection + workflow
> end-to-end (agent → IA2 → Modbus → plant → KPI). The final deployment will run
> on a **Mac mini** connected to the real I/O of the physical 3-tank rig.

### Process & register map

Three-tank benchmark: Tank 1 & Tank 3 independently pumped, each with an SSR
heater, coupled through the middle Tank 2 → the cascade / decoupling problem.

| Register | Addr | Variable | Units | Scale |
| --- | --- | --- | --- | --- |
| 40001 | 0 | Tank 1 Level | m | ×1e-4 |
| 40002 | 1 | Tank 1 Temperature | °C | ×1e-2 |
| 40003 | 2 | Tank 2 Level | m | ×1e-4 |
| 40004 | 3 | Tank 2 Temperature | °C | ×1e-2 |
| 40005 | 4 | Tank 3 Level | m | ×1e-4 |
| 40006 | 5 | Tank 3 Temperature | °C | ×1e-2 |
| 40007 | 6 | Pump 1 drive | frac 0–1 | ×1e-4 |
| 40008 | 7 | Pump 2 drive | frac 0–1 | ×1e-4 |
| 40009 | 8 | Reset command (nonce) | — | ×1 |
| 40010–12 | 9–11 | Init levels (Tank 1–3) | m | ×1e-4 |
| 40013–15 | 12–14 | Heaters 1–3 (SSR duty) | frac 0–1 | ×1e-4 |

Engineering value = raw register × scale. The single source of truth is
[`ia2_config.json`](ia2_config.json).

### Safety model (5 layers)

| Layer | What | Where |
| --- | --- | --- |
| L1–L4 | Hardware (RCD, low/high-level switches, thermal protector) | Physical plant — not modeled |
| **L5** | **Software shield** — clamps + interlocks every actuator request | **`threetank.st` (this repo)** |

The L5 shield runs in the PLC scan loop. The agent writes `actuator*_req` /
`heater*_req`; the PLC body clamps each to `[0, 10000]` and forces the mapped
output to 0 when an action would soon trip hardware:

- **Pumps OFF** above 0.55 m (preempts the high-level overflow switch)
- **Heaters OFF** below 0.05 m (preempts dry-fire) or above 70 °C (preempts the
  thermal protector)

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
| PID | `FB_PID` × 5 | PLC | `*_sp` setpoints |
| MPC | `MPCAgent` (numpy) | Python supervisor | `actuator*_req` |
| NMPC | `NMPCOracle` (CasADi) | Python supervisor | `actuator*_req` |
| RL | Trained SAC/PPO | Python supervisor | setpoints (supervisory) or `actuator*_req` |

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
python3 controllers/benchmark.py --rl controllers/sac_threetank.zip --reward-mode kpi
# add --nmpc for the CasADi NMPC oracle (slow)
```

Output (AIO-Gym-style KPI table, ranked):
```
=== Benchmark (mode=kpi, 14 eps x 200 steps) ===
controller     kpi   ±std temp_err  lvl_cm excess_kwh interlock
---------------------------------------------------------------
RL-SAC       97.13   0.52     0.56    1.83      0.015      0.00
MPC          86.75   1.81     3.17    2.74      0.262      0.00
PID          84.18   2.03     3.80    2.35      0.352      0.00
```

> NMPC is excluded by default (CasADi + IPOPT is ~1–4 s/step, adding ~20 min to
> the benchmark). Include it with `--nmpc`:
> ```bash
> python3 controllers/benchmark.py --rl controllers/sac_threetank.zip --nmpc --reward-mode kpi
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
python3 controllers/benchmark.py --rl controllers/sac_threetank.zip --reward-mode kpi

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

- **5 sliders** (pump/heater duty, 0–100 %, with live % readout)
- **Mode dropdown** — switch between Manual / PID / MPC / RL on the fly
- **Reset button** — new random init levels
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
python3 controllers/benchmark.py --rl controllers/sac_cascade.zip --reward-mode kpi

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
├── ia2_config.json            # single contract — register map, scales, setpoints
├── mock_cabinet.py            # pymodbus TCP plant on :5020 (--time-scale k)
├── aio_bridge_env.py          # Gymnasium env (ia2 / edge / modbus backends; --mode)
├── aio_vec_env.py             # vectorized training env (N cabinets, AsyncVectorEnv)
├── run_mode.sh                # boot + run one controller + teardown (one command)
├── tools/
│   └── gen_ia2_artifacts.py   # contract → device/iomap TOMLs (+ ST VAR check)
├── controllers/
│   ├── threetank_model.py     # numpy 3-tank plant (AIO-Gym model interface)
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
│   └── manual_gui.py          # tkinter manual control GUI
├── tests/
│   ├── smoke_reset.py         # reset snaps levels to targets
│   ├── smoke_heater.py        # heater raises temp; cold pump inflow slows it
│   ├── smoke_env.py           # env reset/step/reward over Modbus
│   └── run_smoke.sh           # one-command runner (boots + tests + teardown)
├── ia2_project/               # IA2 PLC project (IEC 61131-3 ST + device + iomap)
│   ├── devices/mock_cabinet.toml   # AUTO-GENERATED — Modbus TCP device
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
python3 controllers/benchmark.py --rl controllers/sac_threetank.zip

# interactive manual control GUI
./run_mode.sh gui

# run the smoke tests
./tests/run_smoke.sh
```

### Smoke tests

```bash
./tests/run_smoke.sh     # boots the cabinet, runs all three, tears down
```

- `smoke_reset.py` — reset snaps tank levels to requested targets
- `smoke_heater.py` — heater raises temp; cold pump inflow slows it (the cascade)
- `smoke_env.py` — env resets (randomized), steps, and rewards over Modbus

### Deployment (sim → hardware)

IA2 fronts the plant through the device config, not in code. Moving from the local
simulator to a physical PLC is a configuration change: repoint
`ia2_project/devices/mock_cabinet.toml` at the PLC's IP and align its register
addresses/scales to the real I/O map. The PLC program, iomap, and bridge env run
unchanged against real hardware.
