from __future__ import annotations

from datetime import date

import pytest

from app.services import srs


def test_successful_reviews_grow_interval() -> None:
    state = srs.SrsState()
    intervals = []
    for _ in range(4):
        state = srs.review(state, 5)
        intervals.append(state.interval_days)
    assert intervals[:2] == [1, 6] and intervals[2] > 6 and intervals[3] > intervals[2]


def test_failure_resets_and_easiness_has_floor() -> None:
    state = srs.SrsState(repetitions=5, interval_days=30, easiness=1.35)
    state = srs.review(state, 1)
    assert state.repetitions == 0 and state.interval_days == 1
    assert state.easiness == pytest.approx(srs.MIN_EASINESS)


def test_due_date_and_quality_mapping() -> None:
    assert srs.due_date(date(2026, 1, 1), srs.SrsState(interval_days=6)) == date(2026, 1, 7)
    assert srs.quality_from_attempt(True) == 5 and srs.quality_from_attempt(False) == 2
    with pytest.raises(ValueError):
        srs.review(srs.SrsState(), 6)
