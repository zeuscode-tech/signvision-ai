"""Inference in a dedicated thread pool, Top-K and domain filtering."""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

import numpy as np

from ..schemas.manifest import Preprocessing
from .model_loader import LoadedModel, normalize_label
from .preprocessing import build_tensor

P = ParamSpec("P")
T = TypeVar("T")


@dataclass(frozen=True)
class Candidate:
    index: int
    label: str
    confidence: float


@dataclass
class InferenceOutput:
    probabilities: np.ndarray
    inference_ms: float


class ClassFilter:
    """Allowed classes of the active domain pack. `None` labels = every class except no-event."""

    def __init__(
        self,
        model: LoadedModel,
        allowed_labels: list[str] | None = None,
        display_labels: dict[int, str] | None = None,
    ) -> None:
        self.size = len(model.labels)
        mask = np.ones(self.size, dtype=bool)
        if allowed_labels is not None:
            mask[:] = False
            for label in allowed_labels:
                index = model.label_index.get(normalize_label(label))
                if index is not None:
                    mask[index] = True
        if model.no_event_index is not None:
            mask[model.no_event_index] = False
        self.mask = mask
        self.labels = model.labels
        self.display = display_labels or {}

    def label(self, index: int) -> str:
        return self.display.get(index, self.labels[index])

    def top_k(self, probabilities: np.ndarray, k: int = 3) -> list[Candidate]:
        """Top-K among allowed classes. Probabilities are NOT renormalised over the subset:
        the reported confidence is the model's real probability for that class."""
        allowed = np.flatnonzero(self.mask)
        if allowed.size == 0:
            return []
        order = allowed[np.argsort(probabilities[allowed])[::-1][:k]]
        return [Candidate(int(index), self.label(int(index)), float(probabilities[index])) for index in order]


def top_k(probabilities: np.ndarray, labels: list[str], k: int = 3) -> list[Candidate]:
    order = np.argsort(probabilities)[::-1][:k]
    return [Candidate(int(index), labels[int(index)], float(probabilities[index])) for index in order]


def infer_window(model: LoadedModel, frames: list[np.ndarray], config: Preprocessing) -> InferenceOutput:
    started = time.perf_counter()
    tensor = build_tensor(frames, config)
    probabilities = model.probabilities(tensor)
    return InferenceOutput(probabilities=probabilities, inference_ms=(time.perf_counter() - started) * 1000)


class InferenceExecutor:
    """Heavy work runs here so the FastAPI event loop is never blocked."""

    def __init__(self, workers: int) -> None:
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="inference")

    async def infer(
        self, model: LoadedModel, frames: list[np.ndarray], config: Preprocessing
    ) -> InferenceOutput:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, infer_window, model, frames, config)

    async def run(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool, functools.partial(func, *args, **kwargs))

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
