# cascade-control-sandbox

A reinforcement-learning sandbox for **cascade / decoupling control of a 3-tank
process**, built on the [IA2](https://github.com/supcon-international/ia2)
industrial-automation engine and the
[AIO-Gym](https://github.com/supcon-international/AIO-Gym) agent-in-the-loop
pattern. The physical "cabinet" is a Python Modbus TCP simulator; IA2 fronts it
as a real PLC would front real hardware; a Gymnasium environment bridges an RL
agent to IA2.

## Architecture (goal flow)

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


This mirrors the project goal diagram node-for-node:

| Goal-flow node | This repo |
| --- | --- |
| AIO-Gym (Manual / PID / MPC / **RL agent**) | the RL agent + `aio_bridge_env.py` (Gymnasium API) |
| Coding AI Agent ("API wrapper, connects AIO-gym") | `aio_bridge_env.py` — bridges Gym ⇄ IA2 |
| IA2 Platform Automation Engine (IronPLC / Rust) | `ia2/target/release/server` on `127.0.0.1:3001` |
| Control Cabinet Gateway & Modbus I/O | IA2 **iomap** (`ia2_project/`) → Modbus TCP |
| Physical Cascade Hardware (LT/TT, valves/pump/SSR) | `mock_cabinet.py` (extended to 3 tanks) |

## Process & register map

Canonical three-tank benchmark: Tank 1 & Tank 3 are independently pumped
(levels) and each tank has an SSR heater (temperatures), coupled through the
middle Tank 2 → the *cascade / decoupling* control problem. Holding registers
(PLC `4xxxx` → 0-based address):

| Register | Addr | Variable | Units | Scale |
| --- | --- | --- | --- | --- |
| 40001 | 0 | Tank 1 Level | m | ×1e-4 |
| 40002 | 1 | Tank 1 Temperature | °C | ×1e-2 |
| 40003 | 2 | Tank 2 Level | m | ×1e-4 |
| 40004 | 3 | Tank 2 Temperature | °C | ×1e-2 |
| 40005 | 4 | Tank 3 (Decoupled) Level | m | ×1e-4 |
| 40006 | 5 | Tank 3 (Decoupled) Temperature | °C | ×1e-2 |
| 40007 | 6 | Actuator 1 (pump 1 drive) | frac 0–1 | ×1e-4 |
| 40008 | 7 | Actuator 2 (pump 2 drive) | frac 0–1 | ×1e-4 |
| 40009 | 8 | Reset command (nonce) | — | ×1 |
| 40010 | 9 | Init level — Tank 1 | m | ×1e-4 |
| 40011 | 10 | Init level — Tank 2 | m | ×1e-4 |
| 40012 | 11 | Init level — Tank 3 | m | ×1e-4 |
| 40013 | 12 | Heater 1 (SSR duty, Tank 1) | frac 0–1 | ×1e-4 |
| 40014 | 13 | Heater 2 (SSR duty, Tank 2) | frac 0–1 | ×1e-4 |
| 40015 | 14 | Heater 3 (SSR duty, Tank 3) | frac 0–1 | ×1e-4 |

Sensors (40001–40006) are read by IA2 from the cabinet. **Actuators** are written
by the agent each step: pumps (40007–40008) for levels and heaters (40013–40015,
SSR duty) for temperatures — together they form the **cascade** problem (outer
temperature loop ↔ inner level/flow loop), coupled because the thermal mass
`m = ρ·A·h` depends on level and cold pump inflow disturbs temperature. The reset
block (40009–40012) is written by the env between episodes: a fresh nonzero value
in `reset_cmd` snaps the plant to the `init_h*` levels (sampled per episode) and
holds them until `reset_cmd` returns to 0 — giving RL training a controllable
initial-state distribution. Engineering value = raw register × scale. The single
source of truth for this contract is [`ia2_config.json`](ia2_config.json).

**Timing budget (review G6):** observations come from the runtime snapshot
(~10 Hz broadcast cache) on top of the 50 ms Modbus poll and 50 ms PLC scan, so
expect ~200 ms of lag — fine at `control_dt ≥ 0.3 s`; the budget is recorded in
the contract's `timing` block. The device file uses the current
`[transport] kind = "tcp"` schema.

## Safety model

The sandbox assumes the project's 5-layer safety architecture:

- **L1–L4 are hardware** (not modeled here): 30 mA RCD on heater/pump circuits
  (L1); a low-level switch per stage wired in series with heater/pump power for
  dry-fire protection (L2); a high-level switch + passive overflow standpipe
  (L3); the heater's built-in thermal protector (L4). These are the real,
  PLC-independent backstop and the precondition for unattended operation.
- **L5 is the software shield** — what `threetank.st` implements. The agent
  writes `actuator*_req` / `heater*_req` (PLC-internal, API-written); the PLC
  body clamps each request and forces the mapped output to 0 when an action
  would soon trip hardware — pumps OFF above 0.55 m (preempt L3), heaters OFF
  below 0.05 m (preempt L2 dry-fire) or above 70 °C (preempt L4). L5 is an
  efficiency layer: it reduces how often L2–L4 trip (and need manual reset)
  during RL exploration; it is **not** the safety backstop itself.

## Control modes

`aio_bridge_env.py --mode {manual,pid,mpc,rl}` selects the controller (IA2
backend only; the Modbus training backend drives actuators directly). `mode` is
a PLC-internal variable the `threetank.st` body `CASE`-switches on to pick each
actuator's request source; the L5 shield then clamps/intercepts it regardless of
mode:

| Mode | Controller | Agent writes | PLC body |
|---|---|---|---|
| `manual` | `FB_MANSTATION` (in PLC) | `manual_*` (0–100 %) | passes manual out, bumpless to auto |
| `pid` | `FB_PID` ×5 (in PLC) | `*_sp` setpoints | PID tracks the setpoint → actuator |
| `mpc` | external supervisor | `actuator*_req` | passes the request through |
| `rl` | external (`aio_bridge_env`) | `actuator*_req` | passes the request through |

Manual/PID run natively in the PLC (IA2 `library/process-control` FBs); MPC/RL
are external supervisors. `FB_MANSTATION` is always in the chain so mode
switches are bumpless. (The `mpc` supervisor + the vectorized training track are
Phase-3 steps 3b/3c.)

## Repository layout

```
cascade-control-sandbox/
│
├── ia2_config.json        # the single contract — register map, scales, setpoints
├── mock_cabinet.py        # Phase 3 · pymodbus TCP plant on :5020 (reads the contract)
├── aio_bridge_env.py      # Phase 5 · Gymnasium env (ia2 / edge / modbus backends)
├── tools/
│   └── gen_ia2_artifacts.py   # contract -> ia2_project device/iomap TOMLs (+ ST VAR check)
├── tests/                # smoke tests — ./tests/run_smoke.sh boots the cabinet + runs them
│   ├── smoke_reset.py        # reset_cmd + init_h* snap levels to targets
│   ├── smoke_heater.py       # heater raises temp; cold pump inflow slows it
│   ├── smoke_env.py          # env reset/step/reward over the modbus backend
│   └── run_smoke.sh          # one-command runner
├── ia2_project/           # Phase 5 · IA2 project (PLC + device + iomap + tasks)
│   ├── project.toml
│   ├── devices/
│   │   └── mock_cabinet.toml   # AUTO-GENERATED — Modbus TCP device, 15 channels
│   ├── iomap.toml              # AUTO-GENERATED — variable ⇄ channel bindings
│   ├── tasks.toml              # 50 ms cyclic task running ThreeTank
│   └── pous/
│       ├── threetank.st        # IEC 61131-3 ST: mode selector (CASE) + L5 shield
│       ├── fb_pid.st           # vendored from ia2/library/process-control (FB_PID)
│       └── fb_manstation.st    # vendored from ia2/library/process-control (FB_MANSTATION)
├── ia2/                   # Vendored IA2 engine — gitignored; clone separately (Setup step 1)
├── requirements.txt       # Python deps (pymodbus, gymnasium, numpy)
├── run_demo.sh            # Boot everything + run a random-policy rollout
├── .gitignore
└── README.md
```

## Setup (WSL2 / Linux)

```bash
# 1. clone + build IA2 (one-time; build ~10–15 min). ia2/ is gitignored, so a
#    fresh checkout of THIS repo won't include it — vendor it explicitly:
git clone --recursive https://github.com/supcon-international/ia2 ia2
cd ia2 && cargo build --release && cd ..
ls ia2/target/release/cs ia2/target/release/server    # verify binaries

# 2. Python deps (no sudo needed)
pip3 install --user -r requirements.txt

# 3. The IA2 device/iomap TOMLs are generated from ia2_config.json. Regenerate
#    after any contract edit (also used as a CI drift check):
python3 tools/gen_ia2_artifacts.py          # regenerate
python3 tools/gen_ia2_artifacts.py --check  # exit 1 if committed TOMLs are stale
```

## Run

```bash
# Full chain — IA2 backend (RL agent -> Gym -> IA2 -> iomap -> Modbus -> plant)
./run_demo.sh                 # or:  STEPS=40 ./run_demo.sh ia2

# Direct Modbus backend (skip IA2; for quick standalone tests)
./run_demo.sh modbus

# Edge backend (project deployed on a remote edge runtime — G4 route shape):
python3 aio_bridge_env.py --backend edge:<edge_name>   # needs a registered, deployed edge (cs edge / cs deploy)
```

Manually, the pieces are:

```bash
python3 mock_cabinet.py                          # plant on 127.0.0.1:5020
ia2/target/release/server --bind 127.0.0.1:3001  # IA2 HTTP API
ia2/target/release/cs project open ia2_project   # load the PLC project
ia2/target/release/cs run                        # compile + start the scan loop
curl -s http://127.0.0.1:3001/api/runtime/snapshot   # live variable values
python3 aio_bridge_env.py --backend ia2          # RL rollout
```

> Only **one** controller may drive the cabinet at a time. The IA2 backend
> starts IA2 (which owns the actuator registers via the iomap); the Modbus
> backend talks to the cabinet directly — don't run both writes simultaneously.

## Smoke tests

Self-contained checks that the simulated components behave — for CI and for
anyone cloning the repo. They read register addresses/scales from the contract
(no hardcoding) and need only `mock_cabinet.py`:

```bash
./tests/run_smoke.sh          # boots the cabinet, runs all three, tears down
```

- `tests/smoke_reset.py` — `reset_cmd` + `init_h*` snap tank levels to requested targets (two different targets).
- `tests/smoke_heater.py` — a heater raises its tank's temperature; cold pump inflow slows it (the cascade disturbance).
- `tests/smoke_env.py` — the Gym env resets (randomized init levels), steps, and rewards over the Modbus backend.

(The full IA2-in-the-loop chain is exercised by [`./run_demo.sh`](run_demo.sh).)

## Verification

- **Phase 2** — `cs 0.0.1` and `ia2-server` built (`ia2/target/release/`).
- **Phase 3** — `mock_cabinet.py` smoke-tested: writing pump commands raises the
  levels (Tank1/Tank3) and the coupling fills Tank2; temps relax toward supply.
- **Phase 5 (Modbus backend)** — `aio_bridge_env.py` Gym loop (reset/step/reward)
  drives the plant directly; actions → level changes → reward.
- **Phase 5 (IA2 backend)** — the **full chain** verified: env reads
  `/api/runtime/snapshot` and writes `/api/runtime/variables/{name}`; IA2's iomap
  bridges to the cabinet over Modbus TCP; the agent's actions move the levels and
  the reward climbs toward the setpoints.
- **Contract codegen (review item 5)** — `tools/gen_ia2_artifacts.py` regenerates
  the IA2 device/iomap TOMLs from `ia2_config.json` (`--check` is drift-clean), and
  `mock_cabinet.py` is config-driven too — so the register map is single-sourced
  across the cabinet, the iomap, the ST, and the env.
- **Edge backend (review item 4 / G4)** — `--backend edge:<name>` targets an
  edge-deployed project via the dev-server proxy (`GET /api/edges/{name}/status`,
  `POST /api/edges/{name}/runtime/write` body `{name,value}`). Route shapes
  verified against the IA2 source and confirmed by a route-hit; live end-to-end
  validation is pending a real edge deployment.
- **Episode reset (review item 1 / G1)** — `reset_cmd` (40009) + `init_h1/2/3`
  (40010–40012) give the env a controllable initial-state distribution: `reset()`
  samples levels, writes them, and pulses a nonce on `reset_cmd`; the cabinet
  snaps to the init levels and holds. Verified end-to-end through IA2 (3 episodes,
  randomized init levels applied exactly) and via direct Modbus.
- **Heated tanks / cascade loop (review item 2 / G2)** — `heater1/2/3`
  (40013–40015, SSR duty) + a first-law thermal model
  `m·c_p·dT/dt = Q_heat − Q_loss + Σq_in·(T_src − T)` with level-coupled thermal
  mass (`m = ρ·A·h`) make the repo name true: the agent controls 2 pumps (levels)
  + 3 heaters (temps) against the level→temp coupling, and the reward tracks
  both. Verified: a heater raises its tank's temperature and a cold pump inflow
  slows it (the cascade disturbance); a full 5-action rollout runs through IA2.
- **Timing budget + transport (review item 6 / G6)** — the contract states the
  observation-lag budget (~200 ms: 50 ms Modbus poll + 50 ms PLC scan + ~10 Hz
  snapshot cache; `min_control_dt` 0.3 s), and the device TOML uses the current
  `[transport] kind = "tcp"` schema. Verified: `cs project check` + a full chain run.
- **L5 software shield (review item 3)** — `threetank.st` is now a safety layer,
  not pass-through: the agent writes `actuator*_req` / `heater*_req` and the PLC
  body clamps + interlocks (high-level pump cutoff; low-level dry-fire + high-temp
  heater cutoff) before assigning the mapped outputs. Verified: driving pump1 full
  past the 0.55 m cutoff forces `actuator1 = 0` while `actuator1_req` stays 10000
  (preempts the L3 hardware trip).
- **Control-mode selector (Phase 3 / 3a)** — `--mode {manual,pid,mpc,rl}` selects
  the controller via a PLC `mode` variable; `threetank.st` `CASE`-switches the
  request source (Manual/PID via vendored `FB_PID`/`FB_MANSTATION`; MPC/RL via
  `actuator*_req`) and the L5 shield clamps it. Verified: PID tracks the config
  setpoints (reward −2.6→−0.6), manual follows the operator output; the smoke
  suite still passes. (MPC supervisor + vectorized training track = 3b/3c.)

## Workflow integration & deployment

- **Agent tooling** — the IA2 agent skill is bundled at
  [`.claude/skills/industrial-automation-skill`](.claude/skills/industrial-automation-skill);
  load it into Claude Code / Cursor to drive the whole stack — author/compile/run
  PLC programs, force variables, debug — through `cs` and the HTTP API.
- **End-to-end testing** — the validated data flow is:
  `Agent decision → AIO-Gym (Gymnasium) call → IA2 runtime → Modbus TCP →
  simulated tank response` (see *Architecture* and *Verification* above).
- **Deployment readiness (sim → hardware)** — the simulation is *hardware-ready*.
  IA2 fronts the plant through the device config rather than in code, so moving
  from the local simulator to a physical PLC is a configuration change: repoint
  `ia2_project/devices/mock_cabinet.toml` at the PLC's IP **and** align its
  register addresses/scales to the real I/O map (e.g. the LT101/201, TT101/201
  sensors and valve/pump/SSR actuators). The PLC program, iomap, and
  `aio_bridge_env.py` then run unchanged against real hardware.
