import numpy as np
import pytest
from app.ml.features import FEATURE_SIZE, extract_features, normalize_hand


def hand(offset=0.0):
    points = []
    for i in range(21):
        points.extend([offset + i * 0.01, 0.2 + i * 0.005, i * -0.002])
    return points


def test_normalization_is_translation_invariant():
    first = normalize_hand(np.array(hand(0.0)))
    second = normalize_hand(np.array(hand(0.4)))
    assert np.allclose(first, second, atol=1e-5)


def test_one_and_two_hands_have_fixed_shape():
    one = extract_features(hand() + [0.0] * 63)
    two = extract_features(hand() + hand(0.3))
    assert one.shape == (FEATURE_SIZE,)
    assert two.shape == (FEATURE_SIZE,)
    assert np.count_nonzero(one[63:]) == 0


def test_missing_both_hands_is_rejected():
    with pytest.raises(ValueError, match="Руки не обнаружены"):
        extract_features([0.0] * FEATURE_SIZE)


def test_invalid_length_is_rejected():
    with pytest.raises(ValueError, match="Ожидалось"):
        extract_features([1.0, 2.0])

