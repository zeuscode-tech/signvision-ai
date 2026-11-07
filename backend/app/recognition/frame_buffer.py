"""Per-connection ring buffer of preprocessed frames with bounded memory."""

from __future__ import annotations

from collections import deque

import numpy as np


class FrameBuffer:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._frames: deque[np.ndarray] = deque(maxlen=capacity)
        self.frames_since_inference = 0

    def append(self, frame: np.ndarray) -> None:
        self._frames.append(frame)
        self.frames_since_inference += 1

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def is_full(self) -> bool:
        return len(self._frames) == self.capacity

    def ready_for_inference(self, stride: int) -> bool:
        return self.is_full and self.frames_since_inference >= stride

    def snapshot(self) -> list[np.ndarray]:
        """Frames in chronological order; resets the stride counter."""
        self.frames_since_inference = 0
        return list(self._frames)

    def clear(self) -> None:
        self._frames.clear()
        self.frames_since_inference = 0


class DropOldestQueue:
    """Bounded FIFO of raw frames: when full, the oldest unprocessed frame is discarded."""

    def __init__(self, maxlen: int) -> None:
        self._items: deque[tuple[bytes, int]] = deque()
        self.maxlen = maxlen
        self.dropped = 0

    def put(self, item: tuple[bytes, int]) -> None:
        if len(self._items) >= self.maxlen:
            self._items.popleft()
            self.dropped += 1
        self._items.append(item)

    def get(self) -> tuple[bytes, int] | None:
        return self._items.popleft() if self._items else None

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()
