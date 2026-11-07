"""Adapted SM-2 spaced repetition (pure functions, unit-tested)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

MIN_EASINESS = 1.3


@dataclass(frozen=True)
class SrsState:
    repetitions: int = 0
    interval_days: int = 0
    easiness: float = 2.5


def review(state: SrsState, quality: int) -> SrsState:
    """Apply one review with quality 0..5 (SM-2: < 3 means the item was not recalled)."""
    if not 0 <= quality <= 5:
        raise ValueError("quality must be in 0..5")
    if quality < 3:
        repetitions, interval = 0, 1
    else:
        repetitions = state.repetitions + 1
        if repetitions == 1:
            interval = 1
        elif repetitions == 2:
            interval = 6
        else:
            interval = max(1, round(state.interval_days * state.easiness))
    easiness = state.easiness + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    return SrsState(repetitions=repetitions, interval_days=interval, easiness=max(MIN_EASINESS, easiness))


def due_date(today: date, state: SrsState) -> date:
    return today + timedelta(days=state.interval_days)


def quality_from_attempt(success: bool) -> int:
    return 5 if success else 2
