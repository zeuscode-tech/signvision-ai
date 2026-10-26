from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps, UnidentifiedImageError


CLASSES = [
    "Нет жеста", "Ё", "А", "Б", "В", "Г", "Д", "Е", "Ж", "З", "И", "Й",
    "К", "Л", "М", "Н", "О", "П", "Р", "С", "Т", "У", "Ф", "Х", "Ц",
    "Ч", "Ш", "Щ", "Ъ", "Ы", "Ь", "Э", "Ю", "Я",
]
MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def _decode_frame(value: str) -> np.ndarray:
    try:
        encoded = value.split(",", 1)[1] if "," in value else value
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) > 2_000_000:
            raise ValueError("Один кадр превышает лимит 2 МБ")
        with Image.open(BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            # Matches the official demo: aspect-preserving resize, 114 padding.
            image = ImageOps.pad(
                image, (224, 224), method=Image.Resampling.BILINEAR,
                color=(114, 114, 114), centering=(0.5, 0.5),
            )
            return np.asarray(image, dtype=np.float32)
    except (ValueError, IndexError, UnidentifiedImageError) as exc:
        raise ValueError("Некорректный JPEG-кадр") from exc


def preprocess(frames: list[str], window_size: int) -> np.ndarray:
    if len(frames) != window_size:
        raise ValueError(f"Модель Bukva ожидает ровно {window_size} кадров")
    tensors = []
    for value in frames:
        image = (_decode_frame(value) - MEAN) / STD
        tensors.append(np.transpose(image, (2, 0, 1)))
    return np.stack(tensors, axis=0)[None].astype(np.float32)


@lru_cache(maxsize=2)
def load_session(path_text: str) -> ort.InferenceSession:
    return ort.InferenceSession(path_text, providers=["CPUExecutionProvider"])


def model_info(path: Path) -> dict:
    if not path.exists():
        return {"ready": False, "source": "Bukva"}
    session = load_session(str(path))
    model_input = session.get_inputs()[0]
    return {
        "ready": True,
        "source": "Bukva",
        "window_size": int(model_input.shape[1]),
        "input_shape": model_input.shape,
        "classes": CLASSES,
        "license": "Bukva custom CC BY-SA 4.0 variant; see THIRD_PARTY_DATA.md",
        "reported_top1": 0.836,
    }


def predict(path: Path, frames: list[str]) -> dict:
    if not path.exists():
        raise FileNotFoundError("Модель Bukva не установлена")
    session = load_session(str(path))
    model_input = session.get_inputs()[0]
    tensor = preprocess(frames, int(model_input.shape[1]))
    raw = np.asarray(session.run(None, {model_input.name: tensor})[0]).reshape(-1)
    if raw.size != len(CLASSES):
        raise RuntimeError(f"Неожиданный выход модели: {raw.size} классов")
    if np.all(raw >= 0) and np.isclose(float(raw.sum()), 1.0, atol=1e-3):
        probabilities = raw
    else:
        shifted = raw - raw.max()
        exp = np.exp(shifted)
        probabilities = exp / exp.sum()
    order = np.argsort(probabilities)[::-1][:3]
    best = int(order[0])
    confidence = float(probabilities[best])
    # The authors publish Top-1 accuracy, but not a calibrated open-set threshold.
    # Use a conservative local rejection rule instead of always claiming a letter.
    accepted = confidence >= 0.45
    return {
        "label": CLASSES[best] if accepted else None,
        "candidate": CLASSES[best],
        "confidence": confidence,
        "accepted": accepted,
        "is_no_event": best == 0 and accepted,
        "top3": [{"label": CLASSES[int(index)], "confidence": float(probabilities[index])} for index in order],
        "model": "Bukva MobileNetV2-TSM",
        "note": "Порог 45% — консервативное локальное правило отказа, а не опубликованная калибровка Bukva.",
    }
