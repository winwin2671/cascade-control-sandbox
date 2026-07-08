#!/usr/bin/env bash
# run_demo.sh — boot the cascade-control sandbox and run a random-policy
# rollout through aio_bridge_env.py.
#
# Full goal flow (see README.md):
#   RL agent -> aio_bridge_env (Gym) -> IA2 (HTTP) -> iomap -> Modbus TCP
#            -> mock_cabinet (plant) -> back via iomap -> snapshot
#
# Usage:
#   ./run_demo.sh              # IA2 backend (the full chain) — default
#   ./run_demo.sh modbus       # direct Modbus backend (skip IA2)
#   STEPS=40 ./run_demo.sh     # more rollout steps
#
# NOTE: only ONE controller may drive the cabinet at a time. The IA2 backend
# starts IA2 (which owns the actuator registers); the Modbus backend does not.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IA2="$ROOT/ia2/target/release"
BACKEND="${1:-ia2}"
STEPS="${STEPS:-20}"

CAB_PID=""; SRV_PID=""
cleanup() {
  [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null || true
  [ -n "$CAB_PID" ] && kill "$CAB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> starting mock_cabinet (Modbus TCP plant on :5020)"
python3 -u "$ROOT/mock_cabinet.py" --log-every 0 &
CAB_PID=$!
sleep 2

if [ "$BACKEND" = "ia2" ]; then
  echo "==> starting ia2-server (:3001)"
  "$IA2/server" --bind 127.0.0.1:3001 &
  SRV_PID=$!
  for _ in $(seq 1 40); do
    curl -sf -m 1 http://127.0.0.1:3001/api/health >/dev/null && break
    sleep 0.5
  done
  curl -sf -m 1 http://127.0.0.1:3001/api/health >/dev/null \
    || { echo "ERROR: ia2-server did not become healthy"; exit 1; }
  echo "==> opening ia2_project and starting ThreeTank"
  "$IA2/cs" project open "$ROOT/ia2_project"
  "$IA2/cs" run
elif [ "$BACKEND" = "modbus" ]; then
  : # cabinet only
else
  echo "unknown backend '$BACKEND' (use: ia2 | modbus)"; exit 2
fi

echo "==> running aio_bridge_env (backend=$BACKEND, steps=$STEPS)"
python3 -u "$ROOT/aio_bridge_env.py" --backend "$BACKEND" --steps "$STEPS"
