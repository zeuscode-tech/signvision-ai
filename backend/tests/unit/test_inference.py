from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.recognition.inference import ClassFilter, infer_window, top_k
from app.recognition.model_loader import LoadedModel, load_model, to_probabilities
from tests.conftest import write_model_dir


@pytest.fixture
def model(tmp_path: Path) -> LoadedModel:
    write_model_dir(tmp_path / "tiny")
    return load_model(tmp_path / "tiny" / "manifest.json", tmp_path, ["CPUExecutionProvider"])


def test_softmax_only_for_logits() -> None:
    logits = np.array([2.0, 1.0, 0.0])
    probs = to_probabilities(logits, "logits")
    assert np.isclose(probs.sum(), 1) and probs.argmax() == 0
    raw = np.array([0.7, 0.2, 0.1])
    assert np.allclose(to_probabilities(raw, "probabilities"), raw)


def test_top3_is_sorted() -> None:
    result = top_k(np.array([0.15, 0.5, 0.3, 0.05]), ["a", "b", "c", "d"], 3)
    assert [c.label for c in result] == ["b", "c", "a"]
    assert result[0].confidence == pytest.approx(0.5)


def test_domain_filter_excludes_labels_and_no_event_without_renormalising(model: LoadedModel) -> None:
    probabilities = np.array([0.05, 0.6, 0.3, 0.05], dtype=np.float32)
    restricted = ClassFilter(model, ["голова", "помогать"])
    result = restricted.top_k(probabilities)
    assert [c.label for c in result] == ["голова", "помогать"]
    assert result[0].confidence == pytest.approx(0.3)  # real model probability, not renormalised
    assert "no" not in [c.label for c in ClassFilter(model).top_k(probabilities, 4)]


def test_display_labels_follow_domain_pack(model: LoadedModel) -> None:
    pack_filter = ClassFilter(model, ["ВРАЧ"], {1: "ВРАЧ"})
    assert pack_filter.top_k(np.array([0.1, 0.8, 0.05, 0.05]))[0].label == "ВРАЧ"


def test_window_inference_returns_probabilities(model: LoadedModel) -> None:
    red = np.zeros((16, 16, 3), dtype=np.uint8)
    red[..., 0] = 255
    output = infer_window(model, [red] * 4, model.manifest.preprocessing)
    assert np.isclose(output.probabilities.sum(), 1, atol=1e-5)
    assert model.labels[int(output.probabilities.argmax())] == "врач"
    assert output.inference_ms >= 0
