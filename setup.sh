#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.11}"
if [[ ! -x "$PYTHON_BIN" ]] && command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.11)"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python 3.11 не найден по пути $PYTHON_BIN. Установите: brew install python@3.11"
  exit 1
fi

download() {  # download URL FILE SHA256
  local url="$1" file="$2" sha="$3"
  if [[ -s "$file" ]] && [[ "$(shasum -a 256 "$file" | cut -d' ' -f1)" == "$sha" ]]; then
    return
  fi
  mkdir -p "$(dirname "$file")"
  curl -fL --retry 3 -o "$file.part" "$url"
  if [[ "$(shasum -a 256 "$file.part" | cut -d' ' -f1)" != "$sha" ]]; then
    rm -f "$file.part"
    echo "Контрольная сумма $file не совпала — файл не сохранён"
    exit 1
  fi
  mv "$file.part" "$file"
}

echo "[1/6] Создаём Python-окружение"
"$PYTHON_BIN" -m venv "$ROOT_DIR/.venv"
"$ROOT_DIR/.venv/bin/pip" install --upgrade pip
"$ROOT_DIR/.venv/bin/pip" install -r "$ROOT_DIR/backend/requirements-dev.txt"

echo "[2/6] Устанавливаем frontend-зависимости"
cd "$ROOT_DIR/frontend"
npm install

echo "[3/6] Копируем MediaPipe WASM для полностью локального запуска"
mkdir -p "$ROOT_DIR/frontend/public/mediapipe"
cp -R "$ROOT_DIR/frontend/node_modules/@mediapipe/tasks-vision/wasm/"* "$ROOT_DIR/frontend/public/mediapipe/"

echo "[4/6] Загружаем официальную модель Hand Landmarker (frontend и серверный детектор рук)"
MODEL_FILE="$ROOT_DIR/frontend/public/models/hand_landmarker.task"
mkdir -p "$(dirname "$MODEL_FILE")"
if [[ ! -s "$MODEL_FILE" ]]; then
  curl -fL --retry 3 -o "$MODEL_FILE" "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
fi

echo "[5/6] Проверяем модель дактильного алфавита Bukva"
download "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/bukva/models/MobileNetV2_TSM.onnx" \
  "$ROOT_DIR/models/bukva/bukva_mobilenetv2_tsm.onnx" \
  "c8bff60a5da17f63d45a4d0fa348b402938d2fb5db8c672cf8b7ca04ec0c3dba"

echo "[6/6] Загружаем модель изолированных жестов Easy Sign S3D (≈39 МБ)"
download "https://raw.githubusercontent.com/ai-forever/easy_sign/main/S3D.onnx" \
  "$ROOT_DIR/models/easy-sign-s3d/s3d.onnx" \
  "860ecb5e5aff91b4709016c2dc4f5744eea53e024f80c0b3b8f0f916f6bdb949"

echo "Готово. Запустите: ./start.sh"
