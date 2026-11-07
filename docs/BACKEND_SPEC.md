# Техническое задание на backend SignBridge

## 1. Цель

Разработать локальный backend для следующего релиза SignVision AI —
**SignBridge Live**. Сервис должен принимать кадры с камеры через WebSocket,
локально распознавать поддерживаемые изолированные жесты РЖЯ, возвращать
результат с оценкой уверенности, хранить сессии и прогресс, а также управлять
моделями, отраслевыми словарями и пользовательским обучением.

Backend не должен передавать кадры, landmarks или персональные данные во
внешние сервисы. Если модель отсутствует или не уверена в результате, система
обязана вернуть честное состояние неопределённости.

Первая версия не является переводчиком непрерывной жестовой речи. Она работает
с изолированными жестами, между которыми пользователь делает короткие паузы.

## 2. Технологии

- Python 3.11;
- FastAPI и Uvicorn;
- WebSocket;
- SQLAlchemy 2 и Alembic;
- SQLite;
- Pydantic;
- ONNX Runtime;
- OpenCV и NumPy;
- MediaPipe при необходимости серверной обработки;
- scikit-learn и joblib для пользовательских статических моделей;
- pytest, Ruff и mypy.

Целевая среда — macOS на Apple Silicon. Сервер по умолчанию должен слушать
только `127.0.0.1` и не зависеть от облачных API.

## 3. Предлагаемая структура

```text
backend/
├── app/
│   ├── api/
│   │   ├── health.py
│   │   ├── models.py
│   │   ├── sessions.py
│   │   ├── lessons.py
│   │   ├── progress.py
│   │   ├── domain_packs.py
│   │   ├── training.py
│   │   └── data.py
│   ├── websocket/
│   │   ├── recognition.py
│   │   └── connection_manager.py
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── exceptions.py
│   │   └── security.py
│   ├── database/
│   ├── db_models/
│   ├── schemas/
│   ├── repositories/
│   ├── services/
│   ├── recognition/
│   │   ├── model_loader.py
│   │   ├── preprocessing.py
│   │   ├── frame_buffer.py
│   │   ├── inference.py
│   │   ├── stabilization.py
│   │   ├── gesture_segmenter.py
│   │   └── phrase_composer.py
│   ├── training/
│   │   ├── dataset.py
│   │   ├── features.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── model_registry.py
│   ├── dependencies.py
│   └── main.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── alembic/
```

API, бизнес-логика, база данных и ML-код должны быть разделены. Роуты не
должны напрямую выполнять SQL или preprocessing модели.

## 4. Конфигурация

Настройки читаются из переменных окружения и имеют безопасные локальные
значения по умолчанию.

```env
APP_HOST=127.0.0.1
APP_PORT=8000
DATABASE_URL=sqlite:///./data/signvision.db

MODEL_ROOT=./models
ACTIVE_MODEL_ID=easy-sign-s3d
DOMAIN_PACK_ROOT=./domain-packs

MAX_FRAME_SIZE_BYTES=1000000
MAX_WEBSOCKET_FPS=12
FRAME_BUFFER_SIZE=32

DEFAULT_CONFIDENCE_THRESHOLD=0.65
DEFAULT_STABILITY_COUNT=2
DEFAULT_COOLDOWN_MS=1200

SAVE_RAW_FRAMES=false
LOG_LEVEL=INFO
```

При запуске backend должен проверить конфигурацию, создать необходимые
директории и подключить базу. Отсутствующая модель не должна останавливать весь
сервер: `/health` должен сообщать `model_unavailable`, а остальные локальные
функции должны оставаться доступными.

## 5. REST API

Все методы используют префикс `/api/v1`.

### 5.1 Состояние системы

```http
GET /api/v1/health
GET /api/v1/system/info
```

Пример ответа `/health`:

```json
{
  "status": "ok",
  "database": "connected",
  "active_model": "easy-sign-s3d",
  "model_status": "ready",
  "version": "1.0.0"
}
```

`/system/info` возвращает версии backend и Python, операционную систему,
доступные ONNX Runtime providers, статус базы и активную модель. Абсолютные
пользовательские пути возвращать нельзя.

### 5.2 Реестр моделей

```http
GET    /api/v1/models
GET    /api/v1/models/{model_id}
POST   /api/v1/models/register
POST   /api/v1/models/{model_id}/activate
POST   /api/v1/models/{model_id}/validate
DELETE /api/v1/models/{model_id}
```

Удалять активную модель запрещено. Активация должна быть атомарной: старая
модель остаётся доступной, пока новая полностью не загрузилась и не прошла
тестовый инференс.

### 5.3 Сессии распознавания

```http
POST /api/v1/sessions
GET  /api/v1/sessions
GET  /api/v1/sessions/{id}
POST /api/v1/sessions/{id}/finish
```

### 5.4 Отраслевые словари

```http
GET    /api/v1/domain-packs
GET    /api/v1/domain-packs/{id}
POST   /api/v1/domain-packs
PUT    /api/v1/domain-packs/{id}
DELETE /api/v1/domain-packs/{id}
```

### 5.5 Уроки и прогресс

```http
GET  /api/v1/lessons
GET  /api/v1/lessons/{id}
POST /api/v1/attempts
GET  /api/v1/progress
GET  /api/v1/repetition/next
POST /api/v1/repetition/{gesture_id}/result
```

### 5.6 Training Studio

```http
POST   /api/v1/training/classes
GET    /api/v1/training/classes
DELETE /api/v1/training/classes/{id}

POST   /api/v1/training/sessions
POST   /api/v1/training/samples
GET    /api/v1/training/samples
DELETE /api/v1/training/samples/{id}

POST /api/v1/training/jobs
GET  /api/v1/training/jobs/{id}
GET  /api/v1/training/models
POST /api/v1/training/models/{id}/activate
```

### 5.7 Пользовательские данные

```http
GET    /api/v1/data/export
POST   /api/v1/data/import
DELETE /api/v1/data
```

## 6. Manifest модели

Каждая готовая модель регистрируется через JSON manifest.

```json
{
  "id": "easy-sign-s3d",
  "name": "Easy Sign S3D",
  "version": "1.0",
  "type": "isolated_sign",
  "model_path": "s3d.onnx",
  "labels_path": "labels.json",
  "input_shape": [1, 3, 32, 224, 224],
  "frame_count": 32,
  "confidence_threshold": 0.65,
  "stability_count": 2,
  "cooldown_ms": 1200,
  "preprocessing": {
    "color": "RGB",
    "width": 224,
    "height": 224,
    "normalization": "model_default"
  }
}
```

При регистрации необходимо проверить:

- существование файлов модели и labels;
- корректность ONNX-графа;
- допустимую форму входа;
- соответствие количества labels размеру выхода;
- возможность выполнить тестовый инференс;
- отсутствие путей за пределы разрешённого каталога моделей.

## 7. WebSocket распознавания

Адрес:

```text
ws://127.0.0.1:8000/ws/recognition
```

### 7.1 Начало

Первое сообщение клиента:

```json
{
  "type": "start",
  "mode": "isolated_sign",
  "model_id": "easy-sign-s3d",
  "domain_pack": "medical",
  "client": {
    "frame_width": 640,
    "frame_height": 480
  }
}
```

Ответ backend:

```json
{
  "type": "ready",
  "session_id": "uuid",
  "model_id": "easy-sign-s3d",
  "required_frames": 32,
  "max_fps": 12
}
```

### 7.2 Кадры

- кадры передаются бинарными JPEG-сообщениями;
- размер сообщения ограничивается;
- частота ограничивается;
- устаревшие необработанные кадры отбрасываются;
- очередь не должна бесконечно увеличиваться;
- у каждого соединения собственный frame buffer.

### 7.3 Результат

```json
{
  "type": "prediction",
  "state": "recognized",
  "label": "ПОМОЩЬ",
  "confidence": 0.82,
  "top3": [
    {"label": "ПОМОЩЬ", "confidence": 0.82},
    {"label": "НУЖНО", "confidence": 0.11},
    {"label": "ВРАЧ", "confidence": 0.04}
  ],
  "accepted": true,
  "stable_count": 2,
  "inference_ms": 145,
  "timestamp": "2026-11-07T12:00:00Z"
}
```

Допустимые состояния:

- `waiting`;
- `collecting`;
- `processing`;
- `recognized`;
- `uncertain`;
- `no_hands`;
- `cooldown`;
- `model_unavailable`;
- `error`.

При `stop` или разрыве соединения backend завершает сессию, очищает буфер,
освобождает ресурсы и сохраняет только агрегированные технические метрики.

## 8. Pipeline распознавания

### 8.1 Preprocessing

Для каждого кадра:

1. Проверить размер сообщения и формат.
2. Декодировать JPEG.
3. Найти область рук или пользователя.
4. Сформировать crop с безопасным отступом.
5. Изменить размер согласно manifest.
6. Преобразовать цветовое пространство.
7. Выполнить нормализацию модели.
8. Добавить кадр в кольцевой буфер.
9. После заполнения окна запустить инференс.

Параметры preprocessing нельзя дублировать в разных частях кода: источником
истины служит manifest модели.

### 8.2 Инференс

- ONNX session создаётся один раз при загрузке модели;
- инференс не блокирует event loop FastAPI;
- тяжёлая работа выполняется в выделенном worker/thread pool;
- вероятности вычисляются только в соответствии с реальным выходом модели;
- backend формирует Top-3;
- активный domain pack ограничивает допустимые классы.

### 8.3 Стабилизация

Предсказание принимается, только если:

```text
confidence >= threshold
AND класс стабилен N последовательных инференсов
AND класс разрешён активным domain pack
AND закончился cooldown предыдущего результата
```

Один удерживаемый жест не должен добавляться несколько раз. Для этого нужно
учитывать последний класс, время, исчезновение рук, смену класса и паузу.

### 8.4 Неизвестный жест

При недостаточной уверенности backend не выбирает класс принудительно:

```json
{
  "type": "prediction",
  "state": "uncertain",
  "label": null,
  "confidence": 0.38,
  "top3": [],
  "accepted": false
}
```

## 9. Отраслевые словари

Domain pack хранится в JSON:

```json
{
  "id": "medical",
  "name": "Медицина",
  "version": "1.0",
  "enabled_labels": [
    "БОЛЬ",
    "ГОЛОВА",
    "РУКА",
    "ВРАЧ",
    "ЛЕКАРСТВО",
    "ПОМОЩЬ"
  ],
  "phrase_templates": [
    {
      "tokens": ["БОЛЬ", "ГОЛОВА"],
      "result": "У меня болит голова"
    }
  ],
  "quick_responses": [
    "Где у вас болит?",
    "Вам нужна срочная помощь?"
  ]
}
```

Требования:

- каждый label должен существовать в активной модели;
- пустой или повреждённый pack нельзя активировать;
- шаблоны фраз не генерируются автоматически;
- соответствия должны пройти ручную языковую проверку;
- список распознанных gloss-токенов нельзя выдавать за полноценный перевод.

## 10. База данных

Обязательные таблицы:

### `settings`

Тема, язык, активная модель, активный domain pack, пороги, настройки речи и
даты изменения.

### `model_registry`

Идентификатор, название, версия, тип, пути к модели и labels, preprocessing,
метрики, статус активности и дата создания.

### `recognition_sessions`

Режим, модель, domain pack, время начала и окончания, длительность, число
кадров, предсказаний, принятых результатов, исправлений и средняя задержка.

### `recognition_events`

Предсказанный класс, выбранный пользователем класс, confidence, время
инференса, признаки принятия и исправления. Изображения не сохраняются.

### Учебные таблицы

- `lessons`;
- `lesson_attempts`;
- `srs_items`.

### Training Studio

- `training_classes`;
- `training_sessions`;
- `training_samples`;
- `training_jobs`.

Все даты хранятся в UTC. Foreign keys должны быть включены. Изменения схемы
выполняются через Alembic. Прямые SQL-запросы из API-роутов запрещены.

## 11. Training Studio

Поддерживаемые данные:

- статические landmarks;
- последовательности landmarks;
- отрицательные примеры;
- класс `unknown`;
- изображения только после явного согласия пользователя.

Обучение выполняется фоновым заданием со статусами `queued`, `running`,
`completed` и `failed`. Кадры одной capture session нельзя одновременно
использовать в train и test. При недостатке независимых сессий обучение должно
быть отклонено с понятной ошибкой.

Для каждой версии сохраняются:

- accuracy;
- precision;
- recall;
- macro F1;
- confusion matrix;
- количество классов и сессий;
- параметры preprocessing;
- версия датасета;
- время обучения.

Новая модель не перезаписывает предыдущую. Должны поддерживаться активация и
откат версии.

## 12. Экспорт, импорт и удаление

Экспорт содержит настройки, прогресс, metadata Training Studio и manifest
версии формата. Пользовательские модели включаются опционально.

Импорт должен:

- проверять manifest и версию;
- запрещать path traversal;
- проверять размер файлов;
- не выполнять содержимое архива;
- сначала валидировать данные во временной директории;
- заменять данные только после успешной проверки.

Полное удаление очищает прогресс, сессии, пользовательские samples и модели,
но не удаляет встроенные модели приложения. Операция требует явного
подтверждения.

## 13. Ошибки

Единый формат:

```json
{
  "error": {
    "code": "MODEL_NOT_FOUND",
    "message": "Модель распознавания не найдена",
    "details": null,
    "request_id": "uuid"
  }
}
```

Обязательные коды:

- `VALIDATION_ERROR`;
- `MODEL_NOT_FOUND`;
- `MODEL_INVALID`;
- `MODEL_NOT_READY`;
- `DOMAIN_PACK_NOT_FOUND`;
- `INVALID_FRAME`;
- `FRAME_TOO_LARGE`;
- `RATE_LIMIT_EXCEEDED`;
- `DATABASE_ERROR`;
- `TRAINING_DATA_INSUFFICIENT`;
- `TRAINING_FAILED`;
- `IMPORT_FORMAT_INVALID`;
- `INTERNAL_ERROR`.

Stack trace возвращать клиенту нельзя.

## 14. Приватность и безопасность

- принимать соединения только на localhost;
- разрешать CORS только локальному frontend;
- проверять WebSocket Origin;
- ограничивать размер и частоту кадров;
- не сохранять кадры по умолчанию;
- не выводить кадры или полный текст общения в логи;
- не подключать внешнюю аналитику;
- проверять пути и содержимое импортируемых архивов;
- не выполнять пользовательские файлы;
- не хранить медицинские сведения в первой версии.

## 15. Логирование

Разрешено логировать запуск и остановку backend, состояние базы, загрузку и
переключение модели, создание сессии, длительность инференса, ошибки WebSocket,
обучение, экспорт и импорт.

Запрещено логировать кадры, содержимое обучающих файлов, полный текст общения и
персональные или медицинские сведения. Логи должны иметь ротацию и `request_id`.

## 16. Тестирование

### Unit-тесты

- preprocessing и повреждённый JPEG;
- отсутствие руки;
- frame buffer;
- confidence threshold;
- Top-3;
- стабилизация;
- cooldown и подавление повторов;
- domain filtering;
- phrase templates;
- SRS;
- feature extraction.

### Интеграционные тесты

- запуск приложения и `/health`;
- SQLite и Alembic migrations;
- загрузка и валидация модели;
- создание и завершение сессии;
- WebSocket start/stop;
- отправка тестовых кадров;
- экспорт, импорт и удаление;
- Training Studio job;
- корректное освобождение ресурсов после разрыва соединения.

Обязательные проверки перед релизом:

```bash
pytest
ruff check .
mypy app
```

## 17. Производительность

- event loop не блокируется инференсом;
- модель загружается один раз;
- устаревшие кадры отбрасываются;
- буферы освобождаются после отключения;
- память не растёт неограниченно;
- сервис выдерживает не менее 30 минут непрерывной локальной работы;
- сохраняются среднее, медианное и p95 времени инференса.

Целевые ориентиры для MacBook M1 Pro:

- ответ REST без ML — до 200 мс;
- результат после завершения изолированного жеста — до 1 секунды;
- восстановление WebSocket — до 5 секунд.

Фактические значения необходимо измерить, а не заявлять заранее.

## 18. Критерии приёмки

Backend считается готовым, если:

- запускается одной командой и автоматически создаёт базу;
- `/health` честно сообщает состояние модели и базы;
- модель загружается из manifest и проходит тестовый инференс;
- WebSocket принимает реальные кадры;
- возвращаются label, confidence и Top-3;
- неуверенное предсказание отклоняется;
- удерживаемый жест не создаёт повторные слова;
- domain pack действительно фильтрует классы;
- сессии и технические метрики сохраняются;
- кадры не сохраняются по умолчанию;
- Training Studio разделяет train/test по независимым сессиям;
- модели можно активировать и откатывать;
- экспорт, импорт и удаление данных работают;
- критические тесты проходят;
- backend выдерживает 30 минут непрерывной работы.

## 19. Порядок реализации

1. Конфигурация, SQLite, Alembic, `/health` и единые ошибки.
2. Реестр моделей, manifest, ONNX loader и тестовый инференс.
3. Preprocessing, frame buffer и WebSocket.
4. Top-3, confidence, стабилизация и подавление повторов.
5. Сессии, recognition events и domain packs.
6. Учебный прогресс и интервальное повторение.
7. Training Studio, фоновые jobs, метрики и версии моделей.
8. Экспорт, импорт, удаление, безопасность и нагрузочные тесты.

## 20. Итоговый поток

```text
Кадры камеры
→ проверка и preprocessing
→ буфер последовательности
→ ONNX-инференс
→ фильтрация domain pack
→ confidence и стабилизация
→ распознанное слово или uncertain
→ сохранение технических метрик
→ ответ frontend
```

Главный принцип: backend не имитирует распознавание. Если модель не загружена,
жест неизвестен или уверенность недостаточна, API возвращает честное состояние
ошибки или неопределённости.
