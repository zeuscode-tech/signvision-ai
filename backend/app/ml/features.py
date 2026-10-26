import numpy as np

HAND_SIZE = 21 * 3
FEATURE_SIZE = HAND_SIZE * 2


def normalize_hand(values: np.ndarray) -> np.ndarray:
    """Wrist-center and scale one hand; preserve zeros for a missing hand."""
    hand = np.asarray(values, dtype=np.float32).reshape(21, 3)
    if np.allclose(hand, 0):
        return np.zeros(HAND_SIZE, dtype=np.float32)
    centered = hand - hand[0]
    scale = float(np.max(np.linalg.norm(centered[:, :2], axis=1)))
    if scale < 1e-6:
        return np.zeros(HAND_SIZE, dtype=np.float32)
    centered /= scale
    return centered.reshape(-1)


def extract_features(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (FEATURE_SIZE,):
        raise ValueError(f"Ожидалось {FEATURE_SIZE} координат, получено {array.size}")
    if not np.isfinite(array).all():
        raise ValueError("Координаты содержат NaN или Infinity")
    left = normalize_hand(array[:HAND_SIZE])
    right = normalize_hand(array[HAND_SIZE:])
    if np.allclose(left, 0) and np.allclose(right, 0):
        raise ValueError("Руки не обнаружены")
    return np.concatenate([left, right])

