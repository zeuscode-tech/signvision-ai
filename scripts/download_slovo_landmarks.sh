#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="$ROOT_DIR/datasets/slovo"
LANDMARKS="$DEST_DIR/slovo_mediapipe.json"
ANNOTATIONS="$DEST_DIR/annotations.csv"
EXPECTED_SIZE=1258257325
ZIP_URL="https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo.zip"
LANDMARKS_URL="https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo_mediapipe.json"

mkdir -p "$DEST_DIR"
if [[ ! -s "$ANNOTATIONS" ]]; then
  echo "Извлекаем annotations.csv из официального ZIP по HTTP Range (без скачивания 16 ГБ видео)…"
  "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/fetch_zip_member.py" "$ZIP_URL" annotations.csv "$ANNOTATIONS"
fi

if [[ "${1:-}" == "--annotations-only" ]]; then
  echo "Аннотации готовы: $ANNOTATIONS"
  exit 0
fi

echo "Загружаем Slovo MediaPipe landmarks (~1,2 ГБ). Прерванную загрузку можно продолжить повторным запуском."
curl -fL --retry 5 -C - -o "$LANDMARKS" "$LANDMARKS_URL"
ACTUAL_SIZE=$(stat -f%z "$LANDMARKS" 2>/dev/null || stat -c%s "$LANDMARKS")
if [[ "$ACTUAL_SIZE" != "$EXPECTED_SIZE" ]]; then
  echo "Ошибка: ожидалось $EXPECTED_SIZE байт, получено $ACTUAL_SIZE"
  exit 1
fi
echo "Slovo готов: $LANDMARKS"

