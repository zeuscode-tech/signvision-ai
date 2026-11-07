"""Train a Random Forest on landmark features with a session-level holdout."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .dataset import Dataset, check_sufficiency, group_split
from .evaluator import evaluate
from .features import FEATURE_VERSION, feature_size


@dataclass
class TrainedModel:
    estimator: RandomForestClassifier
    metrics: dict[str, Any]


def _forest(n_estimators: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced", random_state=42, n_jobs=-1
    )


def train(dataset: Dataset) -> TrainedModel:
    check_sufficiency(dataset)
    started = time.perf_counter()
    train_idx, test_idx = group_split(dataset)
    labels = dataset.labels
    evaluation_model = _forest(240)
    evaluation_model.fit(dataset.X[train_idx], dataset.y[train_idx])
    report = evaluate(dataset.y[test_idx], evaluation_model.predict(dataset.X[test_idx]), labels)
    # Metrics come from the held-out sessions; the shipped model then uses all sessions.
    final_model = _forest(300)
    final_model.fit(dataset.X, dataset.y)
    report.update(
        {
            "split": "stratified_holdout_by_capture_session",
            "train_samples": int(len(train_idx)),
            "test_samples": int(len(test_idx)),
            "train_sessions": int(len(set(dataset.groups[train_idx].tolist()))),
            "test_sessions": int(len(set(dataset.groups[test_idx].tolist()))),
            "class_count": len(labels),
            "session_count": int(len(set(dataset.groups.tolist()))),
            "negative_labels": sorted(dataset.negative_labels),
            "samples_per_class": dataset.samples_per_class,
            "sessions_per_class": dataset.sessions_per_class,
            "dataset_version": dataset.version,
            "preprocessing": {
                "feature_version": FEATURE_VERSION,
                "kind": dataset.kind,
                "feature_size": feature_size(dataset.kind),
                "normalization": "wrist_centered_hand_scale",
            },
            "training_seconds": round(time.perf_counter() - started, 3),
        }
    )
    return TrainedModel(estimator=final_model, metrics=report)


def predict_proba(estimator: RandomForestClassifier, features: np.ndarray) -> tuple[list[str], np.ndarray]:
    probabilities = estimator.predict_proba(features.reshape(1, -1))[0]
    return [str(label) for label in estimator.classes_], np.asarray(probabilities, dtype=np.float32)
