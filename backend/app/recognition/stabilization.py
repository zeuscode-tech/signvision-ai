"""Acceptance rule and duplicate suppression.

A prediction is accepted only if:
    confidence >= threshold
    AND the class is stable for N consecutive inferences
    AND the class is allowed by the active domain pack (enforced by ClassFilter)
    AND the cooldown after the previous accepted result has elapsed.

A held gesture is accepted once: after acceptance the label is "latched" and
can only be accepted again after a release — hands disappeared, a different
class became stable, the model reported its no-event class, or the latched
class was not seen confidently for `pause_ms`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

State = Literal["recognized", "uncertain", "cooldown"]
Reason = Literal["accepted", "low_confidence", "awaiting_stability", "duplicate_suppressed", "cooldown"]


@dataclass(frozen=True)
class StabilizerConfig:
    threshold: float
    stability_count: int
    cooldown_ms: int
    pause_ms: int


@dataclass(frozen=True)
class Decision:
    state: State
    accepted: bool
    stable_count: int
    reason: Reason


class Stabilizer:
    def __init__(self, config: StabilizerConfig) -> None:
        self.config = config
        self._streak_label: str | None = None
        self._streak = 0
        self._latched_label: str | None = None
        self._latched_seen_ms = 0
        self._last_accepted_ms: int | None = None

    @property
    def latched_label(self) -> str | None:
        return self._latched_label

    def release(self) -> None:
        """Gesture boundary: hands gone, no-event or explicit reset."""
        self._latched_label = None
        self._streak_label = None
        self._streak = 0

    def observe(self, label: str | None, confidence: float, now_ms: int) -> Decision:
        cfg = self.config
        if (
            self._latched_label is not None
            and cfg.pause_ms > 0
            and now_ms - self._latched_seen_ms >= cfg.pause_ms
        ):
            self._latched_label = None

        if label is None or confidence < cfg.threshold:
            self._streak_label = None
            self._streak = 0
            return Decision("uncertain", False, 0, "low_confidence")

        if label == self._streak_label:
            self._streak += 1
        else:
            self._streak_label = label
            self._streak = 1

        if label == self._latched_label:
            self._latched_seen_ms = now_ms
            return Decision("recognized", False, self._streak, "duplicate_suppressed")
        if self._streak < cfg.stability_count:
            return Decision("recognized", False, self._streak, "awaiting_stability")
        # A different class is now stable: the previously held gesture has ended.
        self._latched_label = None
        if self._last_accepted_ms is not None and now_ms - self._last_accepted_ms < cfg.cooldown_ms:
            return Decision("cooldown", False, self._streak, "cooldown")
        self._last_accepted_ms = now_ms
        self._latched_label = label
        self._latched_seen_ms = now_ms
        return Decision("recognized", True, self._streak, "accepted")
