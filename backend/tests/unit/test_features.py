from __future__ import annotations

import numpy as np
import pytest

from app.training.features import (
    SEQUENCE_FEATURE_SIZE,
    STATIC_FEATURE_SIZE,
    FeatureError,
    extract_sequence_features,
    extract_static_features,
    normalize_hand,
)


def hand(offset: float = 0.0, scale: float = 1.0) -> list[float]:
    points: list[float] = []
    for i in range(21):
        points.extend([offset + i * 0.01 * scale, 0.2 + i * 0.005 * scale, i * -0.002 * scale])
    return points


def test_normalization_is_translation_and_scale_invariant() -> None:
    base = normalize_hand(np.array(hand()))
    assert np.allclose(base, normalize_hand(np.array(hand(0.4))), atol=1e-5)
    assert np.allclose(base, normalize_hand(np.array(hand(0.1, 2.0))), atol=1e-5)


def test_one_and_two_hands_have_fixed_shape() -> None:
    one = extract_static_features(hand() + [0.0] * 63)
    two = extract_static_features(hand() + hand(0.3))
    assert one.shape == two.shape == (STATIC_FEATURE_SIZE,)
    assert np.count_nonzero(one[63:]) == 0


def test_missing_hands_and_bad_values_are_rejected() -> None:
    with pytest.raises(FeatureError, match="Руки не обнаружены"):
        extract_static_features([0.0] * STATIC_FEATURE_SIZE)
    with pytest.raises(FeatureError, match="Ожидалось"):
        extract_static_features([1.0, 2.0])
    with pytest.raises(FeatureError, match="NaN"):
        extract_static_features([float("nan")] * STATIC_FEATURE_SIZE)


def test_sequence_features_resample_any_length() -> None:
    short = extract_sequence_features([hand(i * 0.01) + [0.0] * 63 for i in range(5)])
    long = extract_sequence_features([hand(i * 0.01) + [0.0] * 63 for i in range(40)])
    assert short.shape == long.shape == (SEQUENCE_FEATURE_SIZE,)


def test_sequence_rules() -> None:
    with pytest.raises(FeatureError):
        extract_sequence_features([hand() + [0.0] * 63] * 2)
    with pytest.raises(FeatureError, match="половине"):
        extract_sequence_features([[0.0] * 126] * 5 + [hand() + [0.0] * 63])
