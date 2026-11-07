# Сторонние данные и модели

## Bukva

В проект включён файл `models/bukva/bukva_mobilenetv2_tsm.onnx`, опубликованный авторами набора **Bukva: Russian Sign Language Alphabet Dataset**.

- Репозиторий: <https://github.com/ai-forever/bukva>
- Авторы: Karina Kvanchiani, Petr Surovtsev, Alexander Nagaev, Elizaveta Petrova, Alexander Kapitanov.
- Публикация: <https://arxiv.org/abs/2410.08675>
- Модель: MobileNetV2-TSM, 34 класса (`no_event` + 33 буквы).
- Заявленный авторами результат на их test split: Top‑1 83,6%.
- Лицензия: пользовательский вариант CC BY-SA 4.0. Официальная русская копия сохранена в `docs/licenses/bukva-slovo-license-ru.pdf`.

Вес не изменялся. SignVision добавляет локальный preprocessing, показ Top‑3 и консервативный порог отказа 45%. Этот порог не является опубликованной авторами калибровкой.

## Easy Sign S3D

`setup.sh` скачивает `models/easy-sign-s3d/s3d.onnx` (≈39 МБ, SHA-256 проверяется) из репозитория **ai-forever/easy_sign**. В Git хранятся только `manifest.json` и `labels.json` (список классов из `RSL_class_list.txt`, без изменений).

- Репозиторий: <https://github.com/ai-forever/easy_sign>
- Модель: S3D, 1598 изолированных жестов РЖЯ + класс `no`, окно 32 кадра.
- Лицензия: Creative Commons Attribution-ShareAlike 4.0 International.

Вес не изменялся. Preprocessing соответствует официальной демонстрации (полный RGB-кадр, 224×224, x/255, NCTHW). Порог 0,65, стабилизация и cooldown — локальные правила SignBridge, а не калибровка авторов. Отраслевой словарь `domain-packs/medical.json` использует только классы этой модели; его соответствия не проверены экспертом РЖЯ (`review_status: unreviewed`).

## Slovo

Загрузчик и импортёр предназначены для официального файла MediaPipe-landmarks набора **Slovo: Russian Sign Language Dataset**.

- Репозиторий: <https://github.com/hukenovs/slovo>
- Публикация: <https://arxiv.org/abs/2305.14527>
- Официальный landmarks: `slovo_mediapipe.json`, 1 258 257 325 байт.
- Лицензия совпадает с сохранённым выше пользовательским вариантом CC BY-SA 4.0.

Текущая локальная рабочая копия может содержать landmarks в `datasets/slovo/slovo_mediapipe.json`. Файл не требуется для готовой модели алфавита и исключён из Git из-за размера около 1,2 ГБ. Скрипт `scripts/download_slovo_landmarks.sh` загружает его напрямую с официального хранилища с поддержкой продолжения.

