"""Dataset assembly, sufficiency rules and a session-level train/test split.

Frames of one capture session are strongly correlated, so a session is never
split between train and test. Training is refused when there are not enough
independent sessions to evaluate honestly.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..db_models import TrainingClass, TrainingSample, TrainingSession
from .features import feature_size

MIN_CLASSES = 2
MIN_SESSIONS_PER_CLASS = 2
MIN_SAMPLES_PER_CLASS = {"static": 6, "sequence": 4}
TEST_FRACTION = 0.25


class InsufficientDataError(ValueError):
    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


@dataclass
class Dataset:
    kind: str
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    negative_labels: set[str]
    version: str
    sessions_per_class: dict[str, int] = field(default_factory=dict)
    samples_per_class: dict[str, int] = field(default_factory=dict)

    @property
    def labels(self) -> list[str]:
        return sorted(set(self.y.tolist()))


def build_dataset(kind: str, rows: list[tuple[TrainingSample, TrainingSession, TrainingClass]]) -> Dataset:
    expected = feature_size(kind)
    features, labels, groups = [], [], []
    negatives: set[str] = set()
    sessions: dict[str, set[int]] = defaultdict(set)
    digest = hashlib.sha256(kind.encode())
    for sample, session, gesture in rows:
        vector = json.loads(sample.features_json)
        if len(vector) != expected:
            continue  # sample from another feature version / kind
        features.append(vector)
        labels.append(gesture.name)
        groups.append(session.id)
        sessions[gesture.name].add(session.id)
        if gesture.is_negative:
            negatives.add(gesture.name)
        digest.update(f"{sample.id}:{session.id}:{gesture.id}:{gesture.name};".encode())
    X = np.asarray(features, dtype=np.float32).reshape(len(features), expected)
    y = np.asarray(labels, dtype=object)
    return Dataset(
        kind=kind,
        X=X,
        y=y,
        groups=np.asarray(groups, dtype=np.int64),
        negative_labels=negatives,
        version=digest.hexdigest()[:16],
        sessions_per_class={name: len(ids) for name, ids in sessions.items()},
        samples_per_class={name: int((y == name).sum()) for name in sessions},
    )


def check_sufficiency(dataset: Dataset) -> None:
    min_samples = MIN_SAMPLES_PER_CLASS[dataset.kind]
    details: dict[str, object] = {
        "kind": dataset.kind,
        "sessions_per_class": dataset.sessions_per_class,
        "samples_per_class": dataset.samples_per_class,
        "min_classes": MIN_CLASSES,
        "min_sessions_per_class": MIN_SESSIONS_PER_CLASS,
        "min_samples_per_class": min_samples,
    }
    if len(dataset.labels) < MIN_CLASSES:
        raise InsufficientDataError(
            "Нужно минимум два класса с образцами, например целевой жест и отрицательный класс «неизвестно»",
            details,
        )
    if all(label in dataset.negative_labels for label in dataset.labels):
        raise InsufficientDataError("Нужен хотя бы один целевой (не отрицательный) класс", details)
    few_sessions = [
        name for name, count in dataset.sessions_per_class.items() if count < MIN_SESSIONS_PER_CLASS
    ]
    if few_sessions:
        raise InsufficientDataError(
            "Для честной проверки нужно минимум две независимые сессии каждого класса: "
            + ", ".join(few_sessions),
            details,
        )
    few_samples = [name for name, count in dataset.samples_per_class.items() if count < min_samples]
    if few_samples:
        raise InsufficientDataError(
            f"Нужно минимум {min_samples} образцов каждого класса: " + ", ".join(few_samples), details
        )


def group_split(dataset: Dataset, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole sessions per class (≥1 in test, ≥1 in train)."""
    rng = np.random.default_rng(seed)
    test_groups: set[int] = set()
    for label in dataset.labels:
        label_groups = np.unique(dataset.groups[dataset.y == label])
        rng.shuffle(label_groups)
        holdout = min(len(label_groups) - 1, max(1, round(len(label_groups) * TEST_FRACTION)))
        test_groups.update(int(group) for group in label_groups[:holdout])
    is_test = np.isin(dataset.groups, list(test_groups))
    train_idx, test_idx = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    overlap = set(dataset.groups[train_idx].tolist()) & set(dataset.groups[test_idx].tolist())
    if overlap:  # defensive: must never happen
        raise RuntimeError("Train/test leakage between capture sessions")
    return train_idx, test_idx
