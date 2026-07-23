#!/usr/bin/env bash
# run_mode.sh — boot the sandbox and run one controller end-to-end.
#
# Boots mock_cabinet (+ ia2-server + the ThreeTank PLC program for the IA2
# modes), runs the chosen controller, then tears everything down on exit.
# One entry point for every run path.
#
# Usage:
#   ./run_mode.sh                # RL (direct actuator*_req through IA2) — default
#   ./run_mode.sh pid            # PLC FB_PID tracks the config setpoints
#   ./run_mode.sh manual         # operator manual_* -> FB_MANSTATION
#   ./run_mode.sh mpc            # external numpy MPC supervisor (run_mpc.py)
#   ./run_mode.sh nmpc           # CasADi+IPOPT NMPC supervisor (slow; ~1-4 s/step)
#   ./run_mode.sh modbus         # direct Modbus backend (no IA2; cabinet only)
#   ./run_mode.sh gui            # interactive manual control GUI (tkinter sliders + live plot)
#   ./run_mode.sh rl [options]   # RL mode supports --algo <sac|ppo> and --train_track <numpy|modbus>
#   ./run_mode.sh pid --steps 40 # more steps (pid/manual/rl/modbus; mpc/nmpc run 40)
#
# pid/manual/mpc/nmpc/rl go through IA2 + the L5 shield; modbus talks to the
# cabinet directly (the fast training path). See README "Control modes".
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IA2="$ROOT/ia2/target/release"
MODE="${1:-rl}"

# Shift the mode off the args if it was provided as the first argument
if [[ $# -gt 0 ]]; then
  shift
fi

# Default RL attributes and Steps (fallback to STEPS env var if set)
ALGO="sac"
TRAIN_TRACK="numpy"
STEPS="${STEPS:-20}"

# Parse optional flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --algo)
      ALGO="${2:-sac}"
      shift 2
      ;;
    --train_track)
      TRAIN_TRACK="${2:-numpy}"
      shift 2
      ;;
    --step|--steps)
      STEPS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 2
      ;;
  esac
done

case "$MODE" in
  pid|manual|rl|mpc|nmpc|modbus|gui) ;;
  *) echo "usage: $0 [pid|manual|rl|mpc|nmpc|modbus|gui] [--algo sac|ppo] [--train_track numpy|modbus] [--steps N]  (got: $MODE)"; exit 2 ;;
esac

# Determine if we need the IA2 server running
NEEDS_IA2=true
if [ "$MODE" = "modbus" ]; then
  NEEDS_IA2=false
fi
if [ "$MODE" = "rl" ] && [ "$TRAIN_TRACK" = "modbus" ]; then
  NEEDS_IA2=false
fi

CAB_PID=""; SRV_PID=""
cleanup() {
  echo
  echo "==> tearing down (cabinet + ia2-server)"
  [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null || true
  [ -n "$CAB_PID" ] && kill "$CAB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> starting mock_cabinet (Modbus TCP plant on :5020)"
python3 -u "$ROOT/mock_cabinet.py" --log-every 0 &
CAB_PID=$!
sleep 2
if ! kill -0 "$CAB_PID" 2>/dev/null; then
  echo "ERROR: mock_cabinet failed to start (port 5020 in use?). Aborting."
  exit 1
fi

if [ "$NEEDS_IA2" = true ]; then
  echo "==> starting ia2-server (:3001)"
  "$IA2/server" --bind 127.0.0.1:3001 &
  SRV_PID=$!
  sleep 1
  # H3 fix: check the server is alive before health-polling (a stale server on
  # :3001 would satisfy the health check while our process already died on bind)
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "ERROR: ia2-server failed to start (port 3001 in use?). Aborting."
    exit 1
  fi
  for _ in $(seq 1 40); do
    curl -sf -m 1 http://127.0.0.1:3001/api/health >/dev/null && break
    sleep 0.5
  done
  curl -sf -m 1 http://127.0.0.1:3001/api/health >/dev/null \
    || { echo "ERROR: ia2-server did not become healthy"; exit 1; }
  echo "==> opening ia2_project and starting ThreeTank"
  "$IA2/cs" project open "$ROOT/ia2_project"
  "$IA2/cs" run
fi

echo "==> running mode=$MODE" $([ "$MODE" = mpc ] || [ "$MODE" = nmpc ] || [ "$MODE" = gui ] || echo "(steps=$STEPS)")
case "$MODE" in
  modbus)
    python3 -u "$ROOT/aio_bridge_env.py" --backend modbus --steps "$STEPS" ;;
  pid|manual)
    python3 -u "$ROOT/aio_bridge_env.py" --backend ia2 --mode "$MODE" --steps "$STEPS" ;;
  rl)
    # Determine policy path and backend based on train_track and algo
    if [ "$TRAIN_TRACK" = "modbus" ]; then
      POLICY="$ROOT/controllers/policies/${ALGO}_cascade.zip"
      BACKEND="modbus"
    else
      POLICY="$ROOT/controllers/policies/${ALGO}_threetank.zip"
      BACKEND="ia2"
    fi

    echo "==> RL config: algo=$ALGO, train_track=$TRAIN_TRACK, policy=$POLICY, backend=$BACKEND"
    if [ -f "$POLICY" ]; then
      python3 -u "$ROOT/controllers/run_rl.py" --policy "$POLICY" \
        --backend "$BACKEND" \
        --action-mode "${RL_ACTION_MODE:-setpoint}" --steps "${STEPS:-40}"
    else
      echo "(no trained policy at $POLICY; running random RL demo)"
      python3 -u "$ROOT/aio_bridge_env.py" --backend "$BACKEND" --mode rl --steps "$STEPS"
    fi
    ;;
  mpc)
    python3 -u "$ROOT/controllers/run_mpc.py" ;;
  nmpc)
    python3 -u "$ROOT/controllers/run_nmpc.py" ;;
  gui)
    python3 -u "$ROOT/controllers/manual_gui.py" ;;
esac