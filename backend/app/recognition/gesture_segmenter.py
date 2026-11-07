"""Gesture boundaries from hand presence.

Isolated signs are separated by short pauses. When hands are absent for
`pause_ms` the gesture is considered finished: the stabiliser is released (the
same word may be accepted again) and inference pauses until hands return.
"""

from __future__ import annotations

from typing import Literal

SegmentEvent = Literal["unknown", "hands", "hands_missing", "pause_started", "idle"]


class GestureSegmenter:
    def __init__(self, pause_ms: int) -> None:
        self.pause_ms = pause_ms
        self._last_hands_ms: int | None = None
        self._missing_since_ms: int | None = None
        self._idle = True

    @property
    def idle(self) -> bool:
        """True while no hands have been seen for at least pause_ms (or never)."""
        return self._idle

    def update(self, present: bool | None, now_ms: int) -> SegmentEvent:
        if present is None:
            self._idle = False
            return "unknown"
        if present:
            self._last_hands_ms = now_ms
            self._missing_since_ms = None
            self._idle = False
            return "hands"
        if self._missing_since_ms is None:
            self._missing_since_ms = now_ms
        if self._idle:
            return "idle"
        if now_ms - self._missing_since_ms >= self.pause_ms:
            self._idle = True
            return "pause_started"
        return "hands_missing"
