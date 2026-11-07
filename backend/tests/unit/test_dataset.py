from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pytest

from app.db_models import TrainingClass, TrainingSample, TrainingSession
from app.training.dataset import InsufficientDataError, build_dataset, check_sufficiency, group_split
from app.training.trainer import train

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def rows(sessions_per_class: dict[str, int], samples_per_session: int = 4):  # type: ignore[no-untyped-def]
    result = []
    rng = np.random.default_rng(1)
    sample_id = session_id = 0
    for class_index, (name, count) in enumerate(sessions_per_class.items()):
        gesture = TrainingClass(
            id=class_index + 1,
            name=name,
            normalized_name=name.casefold(),
            description="",
            is_negative=name == "unknown",
            created_at=NOW,
        )
        for _ in range(count):
            session_id += 1
            session = TrainingSession(
                id=session_id, class_id=gesture.id, note="", image_consent=False, created_at=NOW
            )
            for _ in range(samples_per_session):
                sample_id += 1
                vector = rng.normal(class_index * 3, 0.1, 126).tolist()
                result.append(
                    (
                        TrainingSample(
                            id=sample_id,
                            session_id=session_id,
                            kind="static",
                            features_json=json.dumps(vector),
                            frame_count=1,
                            created_at=NOW,
                        ),
                        session,
                        gesture,
                    )
                )
    return result


def test_split_never_shares_a_session_between_train_and_test() -> None:
    dataset = build_dataset("static", rows({"A": 4, "B": 3, "unknown": 2}))
    train_idx, test_idx = group_split(dataset)
    assert set(dataset.groups[train_idx]).isdisjoint(set(dataset.groups[test_idx]))
    for label in dataset.labels:  # every class present in both parts
        assert label in set(dataset.y[train_idx]) and label in set(dataset.y[test_idx])


def test_insufficient_independent_sessions_are_rejected() -> None:
    with pytest.raises(InsufficientDataError, match="две независимые сессии"):
        check_sufficiency(build_dataset("static", rows({"A": 1, "unknown": 2}, samples_per_session=10)))
    with pytest.raises(InsufficientDataError, match="минимум два класса"):
        check_sufficiency(build_dataset("static", rows({"A": 3})))
    with pytest.raises(InsufficientDataError, match="целевой"):
        check_sufficiency(build_dataset("static", rows({"unknown": 3})))
    with pytest.raises(InsufficientDataError, match="образцов"):
        check_sufficiency(build_dataset("static", rows({"A": 2, "unknown": 2}, samples_per_session=1)))


def test_dataset_version_is_deterministic() -> None:
    first = build_dataset("static", rows({"A": 2, "unknown": 2}))
    assert first.version == build_dataset("static", rows({"A": 2, "unknown": 2})).version
    assert first.version != build_dataset("static", rows({"A": 3, "unknown": 2})).version


def test_training_reports_all_required_metrics() -> None:
    metrics = train(build_dataset("static", rows({"A": 3, "unknown": 3}))).metrics
    for key in (
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "confusion_matrix",
        "class_count",
        "session_count",
        "preprocessing",
        "dataset_version",
        "training_seconds",
    ):
        assert key in metrics
    assert metrics["class_count"] == 2 and metrics["session_count"] == 6
    assert metrics["accuracy"] == 1.0
