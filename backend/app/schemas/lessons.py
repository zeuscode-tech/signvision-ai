from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import Field

from .common import ApiModel


class LessonOut(ApiModel):
    id: str
    level: int
    title: str
    description: str
    material_status: str
    exercises: list[dict[str, Any]]


class AttemptCreate(ApiModel):
    lesson_id: str | None = Field(default=None, max_length=64)
    gesture_id: str = Field(min_length=1, max_length=120)
    mode: Literal["practice", "quiz", "recognition", "lesson"] = "practice"
    success: bool
    confidence: float | None = Field(default=None, ge=0, le=1)
    duration_seconds: int = Field(default=0, ge=0, le=86_400)


class AttemptOut(ApiModel):
    id: int
    lesson_id: str | None
    gesture_id: str
    mode: str
    success: bool
    confidence: float | None
    duration_seconds: int
    created_at: datetime


class SrsItemOut(ApiModel):
    gesture_id: str
    repetitions: int
    interval_days: int
    easiness: float
    due_date: date
    last_quality: int | None
    last_reviewed_at: datetime | None


class DailyProgress(ApiModel):
    date: date
    attempts: int
    successes: int
    seconds: int


class GestureProgress(ApiModel):
    gesture_id: str
    attempts: int
    successes: int
    success_rate: float


class ProgressOut(ApiModel):
    attempts: int
    successes: int
    success_rate: float | None
    practice_seconds: int
    daily: list[DailyProgress]
    gestures: list[GestureProgress]
    reviews_due: int
    srs_items: int


class RepetitionResult(ApiModel):
    # SM-2 quality: 0 = total blackout, 5 = perfect response.
    quality: int = Field(ge=0, le=5)
