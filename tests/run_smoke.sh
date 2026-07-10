#!/usr/bin/env bash
# Run all smoke tests against a freshly-booted mock_cabinet, then tear down.
#
# Usage:
#   ./tests/run_smoke.sh                 # default: python3, 127.0.0.1:5020
#   PYTHON=/usr/bin/python3 ./tests/run_smoke.sh
#   CABINET_PORT=5025 ./tests/run_smoke.sh
#
# The cabinet is the only service needed (the env test uses the modbus backend,
# not IA2). For the full IA2-in-the-loop chain, use ./run_demo.sh instead.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
HOST="${CABINET_HOST:-127.0.0.1}"
PORT="${CABINET_PORT:-5020}"

echo "==> starting mock_cabinet (${HOST}:${PORT})"
"$PY" -u "$ROOT/mock_cabinet.py" --host "$HOST" --port "$PORT" --log-every 0 &
CAB=$!
cleanup() {
    echo
    echo "==> stopping mock_cabinet (pid $CAB)"
    kill "$CAB" 2>/dev/null || true
    wait "$CAB" 2>/dev/null || true
}
trap cleanup EXIT

sleep 1   # let it bind (each test also retries its own connection)

rc=0
for t in smoke_reset smoke_heater smoke_env; do
    echo
    echo "===================== $t ====================="
    "$PY" -u "$ROOT/tests/$t.py" || rc=1
done

echo
if [ "$rc" -eq 0 ]; then
    echo "==> all smoke tests passed"
else
    echo "==> some smoke tests FAILED"
fi
exit "$rc"
