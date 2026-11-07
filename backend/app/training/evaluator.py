from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "labels": labels,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(macro_p), 4),
        "recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": {
            "precision": [round(float(v), 4) for v in precision],
            "recall": [round(float(v), 4) for v in recall],
            "f1": [round(float(v), 4) for v in f1],
            "support": [int(v) for v in support],
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
