"""Landmark feature extraction for Training Studio models.

Static sample: 126 values = Left hand (21 points × xyz) + Right hand; a missing
hand is zeros. Each hand is centred on the wrist and scaled by hand size.
Sequence sample: per-frame normalisation, resampling to a fixed number of
steps, plus mean absolute velocity.
"""

from __future__ import annotations

import numpy as np

HAND_SIZE = 21 * 3
STATIC_FEATURE_SIZE = HAND_SIZE * 2
SEQUENCE_STEPS = 8
SEQUENCE_FEATURE_SIZE = STATIC_FEATURE_SIZE * SEQUENCE_STEPS + STATIC_FEATURE_SIZE
MIN_SEQUENCE_FRAMES = 4
MAX_SEQUENCE_FRAMES = 120
FEATURE_VERSION = "landmarks-v1"


class FeatureError(ValueError):
    pass


def normalize_hand(values: np.ndarray) -> np.ndarray:
    """Wrist-centre and scale one hand; a missing hand stays all zeros."""
    hand = np.asarray(values, dtype=np.float32).reshape(21, 3)
    if np.allclose(hand, 0):
        return np.zeros(HAND_SIZE, dtype=np.float32)
    centered = hand - hand[0]
    scale = float(np.max(np.linalg.norm(centered[:, :2], axis=1)))
    if scale < 1e-6:
        return np.zeros(HAND_SIZE, dtype=np.float32)
    return (centered / scale).reshape(-1).astype(np.float32)


def _frame(values: list[float] | np.ndarray) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (STATIC_FEATURE_SIZE,):
        raise FeatureError(f"Ожидалось {STATIC_FEATURE_SIZE} координат, получено {array.size}")
    if not np.isfinite(array).all():
        raise FeatureError("Координаты содержат NaN или Infinity")
    left, right = normalize_hand(array[:HAND_SIZE]), normalize_hand(array[HAND_SIZE:])
    has_hands = not (np.allclose(left, 0) and np.allclose(right, 0))
    return np.concatenate([left, right]), has_hands


def extract_static_features(values: list[float] | np.ndarray) -> np.ndarray:
    features, has_hands = _frame(values)
    if not has_hands:
        raise FeatureError("Руки не обнаружены")
    return features


def extract_sequence_features(frames: list[list[float]]) -> np.ndarray:
    if not MIN_SEQUENCE_FRAMES <= len(frames) <= MAX_SEQUENCE_FRAMES:
        raise FeatureError(
            f"Последовательность должна содержать {MIN_SEQUENCE_FRAMES}–{MAX_SEQUENCE_FRAMES} кадров"
        )
    normalized = []
    with_hands = 0
    for frame in frames:
        features, has_hands = _frame(frame)
        normalized.append(features)
        with_hands += int(has_hands)
    if with_hands * 2 < len(frames):
        raise FeatureError("Руки обнаружены менее чем в половине кадров последовательности")
    matrix = np.stack(normalized)  # T x 126
    positions = np.linspace(0, len(frames) - 1, SEQUENCE_STEPS)
    lower = np.floor(positions).astype(int)
    upper = np.minimum(lower + 1, len(frames) - 1)
    weight = (positions - lower)[:, None].astype(np.float32)
    resampled = matrix[lower] * (1 - weight) + matrix[upper] * weight
    velocity = np.abs(np.diff(matrix, axis=0)).mean(axis=0)
    return np.concatenate([resampled.reshape(-1), velocity]).astype(np.float32)


def feature_size(kind: str) -> int:
    return STATIC_FEATURE_SIZE if kind == "static" else SEQUENCE_FEATURE_SIZE


def extract(kind: str, landmarks: list[float] | None, sequence: list[list[float]] | None) -> np.ndarray:
    if kind == "static":
        if landmarks is None:
            raise FeatureError("Нет landmarks")
        return extract_static_features(landmarks)
    if sequence is None:
        raise FeatureError("Нет последовательности landmarks")
    return extract_sequence_features(sequence)
