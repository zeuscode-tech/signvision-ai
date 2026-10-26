#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.11}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python 3.11 не найден по пути $PYTHON_BIN. Установите: brew install python@3.11"
  exit 1
fi

echo "[1/5] Создаём Python-окружение"
"$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"
"$ROOT_DIR/.venv/bin/pip" install --upgrade pip
"$ROOT_DIR/.venv/bin/pip" install -r "$ROOT_DIR/backend/requirements-dev.txt"

echo "[2/5] Устанавливаем frontend-зависимости"
cd "$ROOT_DIR/frontend"
npm install

echo "[3/5] Копируем MediaPipe WASM для полностью локального запуска"
mkdir -p "$ROOT_DIR/frontend/public/mediapipe"
cp -R "$ROOT_DIR/frontend/node_modules/@mediapipe/tasks-vision/wasm/"* "$ROOT_DIR/frontend/public/mediapipe/"

echo "[4/5] Загружаем официальную модель Hand Landmarker"
MODEL_FILE="$ROOT_DIR/frontend/public/models/hand_landmarker.task"
if [[ ! -s "$MODEL_FILE" ]]; then
  curl -fL --retry 3 -o "$MODEL_FILE" "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
fi

echo "[5/5] Загружаем предобученную модель русского дактильного алфавита Bukva"
BUKVA_MODEL="$ROOT_DIR/models/bukva_mobilenetv2_tsm.onnx"
mkdir -p "$ROOT_DIR/models"
if [[ ! -s "$BUKVA_MODEL" ]]; then
  curl -fL --retry 3 -o "$BUKVA_MODEL" "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/bukva/models/MobileNetV2_TSM.onnx"
fi

echo "Готово. Запустите: ./start.sh"

