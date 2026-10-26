#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PORT="${SIGNVISION_BACKEND_PORT:-8000}"
FRONTEND_PORT="${SIGNVISION_FRONTEND_PORT:-5173}"

if [[ ! -x "$ROOT_DIR/.venv/bin/python" || ! -d "$ROOT_DIR/frontend/node_modules" || ! -s "$ROOT_DIR/frontend/public/models/hand_landmarker.task" ]]; then
  echo "Зависимости не готовы. Сначала выполните ./setup.sh"
  exit 1
fi

for PORT in "$BACKEND_PORT" "$FRONTEND_PORT"; do
  if command -v lsof >/dev/null && lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Порт $PORT уже занят. Остановите другой процесс или задайте SIGNVISION_BACKEND_PORT / SIGNVISION_FRONTEND_PORT."
    exit 1
  fi
done

cleanup() {
  [[ -n "${BACKEND_PID:-}" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  [[ -n "${FRONTEND_PID:-}" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$ROOT_DIR/backend"
"$ROOT_DIR/.venv/bin/uvicorn" app.main:app --host 127.0.0.1 --port "$BACKEND_PORT" &
BACKEND_PID=$!
cd "$ROOT_DIR/frontend"
export SIGNVISION_BACKEND_ORIGIN="http://127.0.0.1:$BACKEND_PORT"
npm run dev &
FRONTEND_PID=$!

echo "SignVision AI: http://127.0.0.1:$FRONTEND_PORT"
echo "API:           http://127.0.0.1:$BACKEND_PORT/docs"
echo "Для остановки нажмите Ctrl+C"
wait "$BACKEND_PID" "$FRONTEND_PID"

