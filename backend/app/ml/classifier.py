from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def train_classifier(X: np.ndarray, y: np.ndarray, groups: np.ndarray, path: Path) -> dict:
    labels = sorted(set(y.tolist()))
    if len(labels) < 2:
        raise ValueError("Для обучения нужны минимум два класса, включая «неизвестный жест»")
    class_groups = {label: len(set(groups[y == label].tolist())) for label in labels}
    can_evaluate = all(count >= 2 for count in class_groups.values()) and len(set(groups.tolist())) >= 4
    metrics = None
    if can_evaluate:
        # Hold out at least one whole capture session per class. This preserves
        # every label in train and test without leaking adjacent video frames.
        rng = np.random.default_rng(42)
        test_groups: set[int] = set()
        for label in labels:
            label_groups = np.unique(groups[y == label])
            rng.shuffle(label_groups)
            holdout_count = max(1, round(len(label_groups) * 0.25))
            test_groups.update(int(value) for value in label_groups[:holdout_count])
        is_test = np.isin(groups, list(test_groups))
        train_idx, test_idx = np.flatnonzero(~is_test), np.flatnonzero(is_test)
        evaluation_model = RandomForestClassifier(
            n_estimators=240, class_weight="balanced", random_state=42, n_jobs=-1
        )
        evaluation_model.fit(X[train_idx], y[train_idx])
        predicted = evaluation_model.predict(X[test_idx])
        precision, recall, f1, support = precision_recall_fscore_support(
            y[test_idx], predicted, labels=labels, zero_division=0
        )
        metrics = {
            "labels": labels,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1.tolist(),
            "support": support.tolist(),
            "confusion_matrix": confusion_matrix(y[test_idx], predicted, labels=labels).tolist(),
            "test_samples": int(len(test_idx)),
            "split": "stratified_group_holdout_by_capture_session",
        }
    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    artifact = {
        "model": model,
        "labels": labels,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sample_counts": dict(Counter(y.tolist())),
        "metrics": metrics,
        "threshold": 0.68,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return {k: v for k, v in artifact.items() if k != "model"}


def load_artifact(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def predict(artifact: dict, features: np.ndarray) -> dict:
    probabilities = artifact["model"].predict_proba(features.reshape(1, -1))[0]
    best_index = int(np.argmax(probabilities))
    label = str(artifact["model"].classes_[best_index])
    confidence = float(probabilities[best_index])
    accepted = confidence >= float(artifact.get("threshold", 0.68))
    return {
        "label": label if accepted else None,
        "candidate": label,
        "confidence": confidence,
        "accepted": accepted,
        "message": "Распознано" if accepted else "Не удалось уверенно распознать",
    }

