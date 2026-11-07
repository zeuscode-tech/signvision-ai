# SignBridge backend

Локальный backend SignVision AI / SignBridge Live: распознавание изолированных
жестов РЖЯ по потоку кадров через WebSocket, сессии и технические метрики,
отраслевые словари, уроки и интервальное повторение, Training Studio,
экспорт/импорт/удаление данных. Сервер слушает только `127.0.0.1`, не обращается
к облачным API и не сохраняет кадры по умолчанию.

Первая версия — **не переводчик непрерывной жестовой речи**: она распознаёт
изолированные жесты, разделённые короткими паузами. Если модель не загружена,
жест неизвестен или уверенность ниже порога, API возвращает честное состояние
(`model_unavailable`, `uncertain`), а не придуманный ответ.

## Запуск

```bash
./setup.sh      # один раз: venv, зависимости, модели (с проверкой SHA-256)
./start.sh      # backend + frontend
```

Только backend:

```bash
cd backend
../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

При старте backend проверяет конфигурацию, создаёт каталоги, применяет миграции
Alembic (база создаётся автоматически), регистрирует встроенные модели из
`models/*/manifest.json` и в фоне загружает активную модель. Отсутствующая модель
не останавливает сервер: `/api/v1/health` сообщает `model_unavailable`, остальные
функции работают.

Документация OpenAPI: <http://127.0.0.1:8000/docs>.

## Структура

```text
backend/
├── app/
│   ├── api/            REST-роуты (только вызов сервисов) + legacy.py для старого UI
│   ├── websocket/      /ws/recognition и учёт подключений
│   ├── core/           config, logging, exceptions, security, timeutils
│   ├── database/       engine (foreign keys ON), UTC-даты, запуск миграций
│   ├── db_models/      таблицы SQLAlchemy
│   ├── schemas/        Pydantic: API, manifest модели, формат экспорта, WS-сообщения
│   ├── repositories/   весь SQL
│   ├── services/       бизнес-логика
│   ├── recognition/    model_loader, preprocessing, hand_detection, frame_buffer,
│   │                   inference, stabilization, gesture_segmenter, phrase_composer
│   ├── training/       features, dataset, trainer, evaluator, model_registry
│   ├── dependencies.py контейнер сервисов
│   └── main.py
├── alembic/            миграции (0001 также переносит данные старой схемы)
└── tests/              unit/ и integration/
```

## Конфигурация

Переменные окружения или `.env` в корне проекта (см. `.env.example`).
Относительные пути считаются от корня проекта.

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `APP_HOST` / `APP_PORT` | `127.0.0.1` / `8000` | только loopback; другой адрес отклоняется при старте |
| `DATABASE_URL` | `sqlite:///./data/signvision.db` | только SQLite |
| `MODEL_ROOT` | `./models` | встроенные модели и `user/` для Training Studio |
| `ACTIVE_MODEL_ID` | `easy-sign-s3d` | модель по умолчанию, пока пользователь не выбрал другую |
| `DOMAIN_PACK_ROOT` | `./domain-packs` | JSON-словари |
| `ONNX_PROVIDERS` | `CPUExecutionProvider` | можно переопределить в manifest модели |
| `HAND_DETECTOR` | `mediapipe` | `none` — без серверного детектора рук |
| `MAX_FRAME_SIZE_BYTES` | `1000000` | лимит одного кадра |
| `MAX_WEBSOCKET_FPS` | `12` | лишние кадры отбрасываются |
| `FRAME_BUFFER_SIZE` | `32` | верхняя граница окна модели |
| `MAX_WEBSOCKET_CONNECTIONS` | `4` | одновременные подключения |
| `DEFAULT_CONFIDENCE_THRESHOLD` / `DEFAULT_STABILITY_COUNT` / `DEFAULT_COOLDOWN_MS` | `0.65` / `2` / `1200` | если не заданы в manifest или настройках пользователя |
| `PAUSE_MS` | `600` | сколько рук не должно быть в кадре, чтобы жест считался законченным |
| `SAVE_RAW_FRAMES` | `false` | сохранять JPEG-кадры в `data/raw_frames/` (только для отладки) |
| `LOG_LEVEL` | `INFO` | логи в `data/logs/backend.log` с ротацией |
| `ALLOWED_ORIGINS` | `http://127.0.0.1:5173,http://localhost:5173` | CORS и Origin для WebSocket |

Приоритет порогов распознавания: настройки пользователя (`PUT /api/v1/settings`) →
manifest модели → `DEFAULT_*`.

## REST API (`/api/v1`)

| Метод | Путь | Описание |
|---|---|---|
| GET | `/health` | `status`, `database`, `active_model`, `model_status`, `version` |
| GET | `/system/info` | версии, ОС, ONNX providers, статус базы и модели (без абсолютных путей) |
| GET/PUT | `/settings` | тема, язык, активный словарь, пороги, речь |
| GET | `/models` | реестр (`?type=isolated_sign` / `static_landmarks` / `sequence_landmarks`) |
| GET | `/models/{id}` | запись + labels и manifest |
| POST | `/models/register` | `{"manifest_path": "my-model/manifest.json"}` (относительно `MODEL_ROOT`) |
| POST | `/models/{id}/activate` | атомарная активация после тестового инференса |
| POST | `/models/{id}/validate` | отчёт по всем проверкам |
| DELETE | `/models/{id}` | убрать из реестра; активную удалить нельзя (`MODEL_ACTIVE`) |
| POST/GET | `/sessions` | создать / список сессий распознавания |
| GET | `/sessions/{id}` | метрики и события |
| POST | `/sessions/{id}/finish` | завершить |
| GET/POST | `/domain-packs` | список (с проверкой совместимости) / создать |
| GET/PUT/DELETE | `/domain-packs/{id}` | словарь; активный удалить нельзя |
| GET | `/lessons`, `/lessons/{id}` | уроки |
| POST | `/attempts` | попытка + обновление SM-2 |
| GET | `/progress` | итоги, по дням, по жестам, повторения |
| GET | `/repetition/next` | что повторить сегодня |
| POST | `/repetition/{gesture_id}/result` | `{"quality": 0..5}` |
| POST/GET/DELETE | `/training/classes[/{id}]` | классы (`is_negative` для «неизвестно») |
| POST/GET | `/training/sessions` | независимые capture sessions (`image_consent`) |
| POST/GET/DELETE | `/training/samples[/{id}]` | `kind: static` (126 чисел) или `sequence` (4–120 кадров) |
| POST | `/training/jobs` | фоновое обучение → 202; недостаток данных → 422 `TRAINING_DATA_INSUFFICIENT` |
| GET | `/training/jobs/{id}` | `queued` / `running` / `completed` / `failed` + метрики |
| GET | `/training/models` | версии моделей |
| POST | `/training/models/{id}/activate` | активация версии |
| POST | `/training/models/rollback?kind=static` | откат на предыдущую активную версию |
| POST | `/training/predict` | предсказание активной пользовательской моделью |
| GET | `/data/export?include_models=false` | ZIP-архив |
| POST | `/data/import` | multipart `file` |
| DELETE | `/data?confirm=DELETE_ALL_USER_DATA` | полное удаление пользовательских данных |

Старые эндпоинты `/api/*` (без `v1`) оставлены как deprecated-адаптеры поверх
новых сервисов, чтобы текущий frontend продолжал работать.

### Ошибки

```json
{"error": {"code": "MODEL_NOT_FOUND", "message": "Модель распознавания не найдена",
           "details": null, "request_id": "uuid"}}
```

Коды: `VALIDATION_ERROR`, `NOT_FOUND`, `CONFLICT`, `CONFIRMATION_REQUIRED`,
`FORBIDDEN`, `MODEL_NOT_FOUND`, `MODEL_INVALID`, `MODEL_NOT_READY`, `MODEL_ACTIVE`,
`DOMAIN_PACK_NOT_FOUND`, `DOMAIN_PACK_INVALID`, `SESSION_NOT_FOUND`,
`LESSON_NOT_FOUND`, `INVALID_FRAME`, `FRAME_TOO_LARGE`, `RATE_LIMIT_EXCEEDED`,
`DATABASE_ERROR`, `TRAINING_DATA_INSUFFICIENT`, `TRAINING_FAILED`,
`IMPORT_FORMAT_INVALID`, `INTERNAL_ERROR`. Stack trace клиенту не возвращается.
`request_id` совпадает с заголовком `X-Request-ID` и с записью в логе.

## WebSocket `ws://127.0.0.1:8000/ws/recognition`

1. Клиент: `{"type": "start", "mode": "isolated_sign", "model_id": "easy-sign-s3d", "domain_pack": "medical", "client": {"frame_width": 640, "frame_height": 480}}`.
   `model_id` и `domain_pack` необязательны (по умолчанию активная модель и словарь из настроек; `"domain_pack": null` — без словаря). `session_id` позволяет продолжить сессию, созданную через REST.
2. Сервер: `{"type": "ready", "session_id", "model_id", "required_frames", "max_fps", "max_frame_bytes", "domain_pack", "hand_detection", "thresholds"}`.
3. Клиент шлёт бинарные JPEG-кадры. Сервер отвечает сообщениями `{"type": "prediction", "state": ...}`:

| state | Когда |
|---|---|
| `waiting` | сразу после старта; модель видит свой класс «нет жеста» |
| `collecting` | окно ещё не заполнено (`buffered` / `required`) |
| `processing` | запущен инференс |
| `recognized` | уверенный класс; `accepted: true` только при стабильности N инференсов, вне cooldown и не повтор удерживаемого жеста (`reason`: `accepted` / `awaiting_stability` / `duplicate_suppressed`) |
| `uncertain` | уверенность ниже порога: `label: null`, `top3: []` |
| `cooldown` | новый класс стабилен, но не истёк cooldown |
| `no_hands` | детектор не видит рук; после `PAUSE_MS` жест считается законченным, инференс приостанавливается (окно продолжает скользить) |
| `model_unavailable` | модель не загружена; соединение закрывается с кодом 1013 |

   Пример принятого слова:
   `{"type": "prediction", "state": "recognized", "label": "врач", "confidence": 0.82, "top3": [...], "accepted": true, "stable_count": 2, "reason": "accepted", "event_id": 17, "inference_ms": 145, "timestamp": "2026-11-07T12:00:00.000Z"}`.

   После каждого принятого слова приходит `{"type": "phrase", "tokens": [...], "gloss": "...", "template_text": null, "is_translation": false}`. `template_text` появляется только при совпадении с шаблоном словаря.
4. Ошибки: `{"type": "error", "state": "error", "code": "INVALID_FRAME" | "FRAME_TOO_LARGE" | "RATE_LIMIT_EXCEEDED" | ..., "message"}` (не чаще раза в секунду на код).
5. Управление: `{"type": "correct", "event_id": 17, "label": "голова"}`, `{"type": "phrase_undo"}`, `{"type": "phrase_clear"}`, `{"type": "reset"}`, `{"type": "ping"}`.
6. `{"type": "stop"}` → `{"type": "stopped", "session_id", "summary": {...}}`. При `stop`, разрыве или остановке сервера сессия завершается, буферы очищаются, детектор рук закрывается, сохраняются только агрегированные метрики (кадры, предсказания, принятые, исправления, mean/median/p95 инференса).

Каждое соединение имеет свою очередь (2 кадра, старые отбрасываются), кольцевой
буфер, стабилизатор и детектор рук. Декодирование и инференс выполняются в пулах
потоков, event loop не блокируется. Origin должен быть в `ALLOWED_ORIGINS`.

## Manifest модели

```json
{
  "id": "easy-sign-s3d", "name": "Easy Sign S3D", "version": "1.0", "type": "isolated_sign",
  "model_path": "s3d.onnx", "labels_path": "labels.json",
  "input_shape": [1, 3, 32, 224, 224], "frame_count": 32,
  "output": "logits", "no_event_label": "no",
  "confidence_threshold": 0.65, "stability_count": 2, "cooldown_ms": 1200, "inference_stride": 4,
  "preprocessing": {"color": "RGB", "width": 224, "height": 224, "resize": "stretch",
                    "crop": "none", "normalization": "scale_0_1", "layout": "NCTHW"}
}
```

Manifest — единственный источник параметров preprocessing. Поля сверх примера
из ТЗ нужны, потому что встроенные модели различаются: у Bukva вход
`[1, 8, 3, 224, 224]` (`layout: NTCHW`), выход уже вероятности
(`output: probabilities`), letterbox (`resize: pad`) и ImageNet-нормализация.
`normalization`: `scale_0_1`, `imagenet`, `custom` (+`mean`/`std`), `none`.
`crop: hands` — квадратный crop вокруг рук с отступом `crop_margin`.

Регистрация и активация проверяют: схему manifest, что пути не выходят за
`MODEL_ROOT`, наличие файлов, `onnx.checker`, тип и форму входа, совпадение
числа labels с выходом, тестовый инференс на нулевом тензоре и (для
`output: probabilities`) что выход — распределение.

## Отраслевые словари

`domain-packs/<id>.json`: `id`, `name`, `version`, `enabled_labels`,
`phrase_templates` (`tokens` → `result`), `quick_responses`, `review_status`
(`unreviewed` / `reviewed`), `reviewed_by`. Labels сравниваются без учёта регистра
и должны существовать в модели; пустой или повреждённый словарь не активируется.
Вероятности при фильтрации не перенормируются: `confidence` — реальная
вероятность модели для класса.

Встроенный `medical.json` содержит только слова, которые есть в словаре
`easy-sign-s3d`. Слов «боль» и «помощь» из примера ТЗ в модели нет (есть
«болеть», «помогать»). Шаблонов фраз нет: соответствия жестов и фраз должны
пройти ручную проверку экспертом РЖЯ.

## Training Studio

- Признаки: 126 чисел (левая + правая рука, нормализация относительно запястья
  и размера кисти). Последовательности ресемплируются до 8 шагов + средняя скорость.
- Разделение train/test — целыми capture sessions, не меньше одной сессии
  каждого класса в обеих частях. Нужны ≥ 2 класса (хотя бы один не
  отрицательный), ≥ 2 сессии и ≥ 6 образцов (static) / 4 (sequence) на класс,
  иначе `TRAINING_DATA_INSUFFICIENT` с понятным сообщением.
- Метрики версии: accuracy, macro precision/recall/F1, per-class, confusion
  matrix, число классов и сессий, параметры preprocessing, `dataset_version`,
  время обучения. Финальная модель обучается на всех данных.
- Версии хранятся в `models/user/<id>/` и не перезаписываются; активация и
  откат через API. Файл модели загружается, только если совпадают SHA-256 из
  реестра и HMAC-подпись ключом этой установки (`data/secret.key`).
- Изображения сохраняются только для сессий с `image_consent: true`.

## Экспорт, импорт, удаление

ZIP: `manifest.json` (`format`, `format_version: 1`, SHA-256 каждого файла),
`settings.json`, `progress.json`, `training.json` (классы, сессии,
нормализованные признаки; без изображений), опционально `models/<id>/…`.

Импорт: лимит размера при загрузке, только разрешённые имена файлов, запрет
`..`, абсолютных путей, симлинков и зашифрованных файлов, защита от zip-бомб,
распаковка во временный каталог, проверка контрольных сумм и схем, затем
замена данных одной транзакцией. Содержимое архива не выполняется; модели с
чужой подписью пропускаются (`skipped_models`) — их нужно переобучить из
импортированных образцов.

Удаление очищает прогресс, сессии распознавания, данные Training Studio,
пользовательские модели и изображения. Встроенные модели и словари не трогаются.

## Приватность и логирование

Сервер принимает только loopback-клиентов, CORS и WebSocket Origin ограничены
локальным frontend, размер и частота кадров ограничены, кадры не сохраняются
(если не включён `SAVE_RAW_FRAMES`), внешней аналитики нет. В логах — запуск и
остановка, состояние базы, загрузка/переключение модели, сессии, время
инференса, ошибки, обучение, экспорт/импорт. Кадры, landmarks, содержимое
обучающих файлов и распознанный текст не логируются. Ротация: 5 файлов по 5 МБ.

## Проверки

```bash
cd backend
../.venv/bin/pytest
../.venv/bin/ruff check .
../.venv/bin/mypy app
```

Тесты используют крошечную детерминированную ONNX-модель, поэтому не зависят от
загрузки больших весов; тесты реальных моделей пропускаются, если `setup.sh`
ещё не выполнен.

Нагрузочный тест (30 минут, переподключение каждые 2 минуты, замер памяти):

```bash
.venv/bin/python scripts/soak_test.py --minutes 30 --pid "$(pgrep -f 'uvicorn app.main')"
```

## Измеренная производительность

Измерено 2026-11-07 на этом Mac (Apple Silicon, macOS 27, Python 3.11.9,
ONNX Runtime 1.20.1, CPUExecutionProvider), модель `easy-sign-s3d`. Это не
MacBook M1 Pro из ТЗ — на целевой машине значения нужно измерить заново
командой выше.

| Показатель | Результат | Цель ТЗ |
|---|---|---|
| Непрерывная работа | 30 мин, 15 сессий по 2 мин, 12 FPS, 4634 инференса, 0 ошибок в логе | ≥ 30 мин |
| Инференс S3D (окно 32 кадра) | mean 268.6 мс, median 267.7 мс, p95 280.6 мс | — |
| Результат после жеста | окно анализируется каждые 4 кадра (≈333 мс) + ≈270 мс инференса ⇒ ≈0,6 с (оценка; на реальных жестах не измерялось) | ≤ 1 с |
| Переподключение WebSocket до `ready` | median 2.9 мс, max 19.3 мс | ≤ 5 с |
| Потерянные кадры при 12 FPS | 0 | — |
| Память backend (RSS) | 451 → 460 МБ за 30 мин, без роста | без неограниченного роста |
| REST без ML под нагрузкой инференса | median 2–3 мс, max 34 мс | ≤ 200 мс |

Нагрузочный прогон шёл на синтетических кадрах с `HAND_DETECTOR=none`
(инференс на каждом окне — худший случай по CPU). Точность распознавания на
реальных жестах этим тестом не измеряется — см. `docs/MANUAL_CAMERA_TEST.md`.
