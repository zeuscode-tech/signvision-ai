from pathlib import Path
import numpy as np
from app.ml.classifier import load_artifact, predict, train_classifier


def test_train_save_load_and_unknown_rejection(tmp_path: Path):
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal(-1, .08, (20, 126)), rng.normal(1, .08, (20, 126))]).astype("float32")
    y = np.array(["A"] * 20 + ["unknown"] * 20)
    groups = np.array([1]*10 + [2]*10 + [3]*10 + [4]*10)
    path = tmp_path / "model.joblib"
    report = train_classifier(X, y, groups, path)
    assert path.exists()
    assert set(report["labels"]) == {"A", "unknown"}
    artifact = load_artifact(path)
    result = predict(artifact, np.full(126, -1, dtype="float32"))
    assert result["candidate"] == "A"
    assert 0 <= result["confidence"] <= 1


def test_train_requires_two_classes(tmp_path: Path):
    import pytest
    with pytest.raises(ValueError, match="минимум два класса"):
        train_classifier(np.ones((3,126)), np.array(["A"]*3), np.array([1,2,3]), tmp_path/"m.joblib")

