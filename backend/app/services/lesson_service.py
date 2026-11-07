from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from sqlalchemy.orm import Session

from ..core.exceptions import AppError, ErrorCode
from ..core.timeutils import utcnow, utctoday
from ..db_models import Lesson, LessonAttempt, SrsItem
from ..repositories import lesson_repo
from ..schemas.lessons import (
    AttemptCreate,
    DailyProgress,
    GestureProgress,
    LessonOut,
    ProgressOut,
    SrsItemOut,
)
from . import srs
from .lesson_catalog import LEVELS


def seed_lessons(db: Session) -> None:
    """Keep the built-in lesson catalogue in sync with the database."""
    existing = {lesson.id: lesson for lesson in lesson_repo.list_lessons(db)}
    for item in LEVELS:
        values: dict[str, Any] = {
            "level": item["level"],
            "title": item["title"],
            "description": item["description"],
            "material_status": item["material_status"],
            "exercises_json": json.dumps(item["exercises"], ensure_ascii=False),
        }
        lesson = existing.get(str(item["id"]))
        if lesson is None:
            lesson_repo.add_lesson(db, Lesson(id=str(item["id"]), **values))
        else:
            for key, value in values.items():
                if getattr(lesson, key) != value:
                    setattr(lesson, key, value)
    db.commit()


def lesson_out(lesson: Lesson) -> LessonOut:
    return LessonOut(
        id=lesson.id,
        level=lesson.level,
        title=lesson.title,
        description=lesson.description,
        material_status=lesson.material_status,
        exercises=json.loads(lesson.exercises_json or "[]"),
    )


def list_lessons(db: Session) -> list[LessonOut]:
    return [lesson_out(row) for row in lesson_repo.list_lessons(db)]


def get_lesson(db: Session, lesson_id: str) -> LessonOut:
    lesson = lesson_repo.get_lesson(db, lesson_id)
    if lesson is None:
        raise AppError(ErrorCode.LESSON_NOT_FOUND)
    return lesson_out(lesson)


def _apply_review(db: Session, gesture_id: str, quality: int) -> SrsItem:
    item = lesson_repo.get_srs(db, gesture_id)
    today = utctoday()
    if item is None:
        item = lesson_repo.add_srs(
            db, SrsItem(gesture_id=gesture_id, repetitions=0, interval_days=0, easiness=2.5, due_date=today)
        )
    state = srs.review(srs.SrsState(item.repetitions, item.interval_days, item.easiness), quality)
    item.repetitions, item.interval_days, item.easiness = (
        state.repetitions,
        state.interval_days,
        state.easiness,
    )
    item.due_date = srs.due_date(today, state)
    item.last_quality = quality
    item.last_reviewed_at = utcnow()
    return item


def record_attempt(db: Session, payload: AttemptCreate) -> LessonAttempt:
    if payload.lesson_id is not None and lesson_repo.get_lesson(db, payload.lesson_id) is None:
        raise AppError(ErrorCode.LESSON_NOT_FOUND)
    attempt = lesson_repo.add_attempt(db, LessonAttempt(**payload.model_dump()))
    _apply_review(db, payload.gesture_id, srs.quality_from_attempt(payload.success))
    db.commit()
    return attempt


def record_repetition(db: Session, gesture_id: str, quality: int) -> SrsItemOut:
    item = _apply_review(db, gesture_id, quality)
    db.commit()
    return SrsItemOut.model_validate(item)


def next_repetitions(db: Session, limit: int) -> list[SrsItemOut]:
    return [SrsItemOut.model_validate(item) for item in lesson_repo.list_due(db, utctoday(), limit)]


def progress(db: Session) -> ProgressOut:
    attempts = lesson_repo.list_attempts(db)
    daily: dict[Any, dict[str, int]] = defaultdict(lambda: {"attempts": 0, "successes": 0, "seconds": 0})
    per_gesture: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in attempts:
        bucket = daily[row.created_at.date()]
        bucket["attempts"] += 1
        bucket["successes"] += int(row.success)
        bucket["seconds"] += row.duration_seconds
        per_gesture[row.gesture_id][0] += 1
        per_gesture[row.gesture_id][1] += int(row.success)
    total = len(attempts)
    successes = sum(int(row.success) for row in attempts)
    return ProgressOut(
        attempts=total,
        successes=successes,
        success_rate=(successes / total) if total else None,
        practice_seconds=sum(row.duration_seconds for row in attempts),
        daily=[DailyProgress(date=key, **value) for key, value in sorted(daily.items())],
        gestures=[
            GestureProgress(gesture_id=name, attempts=a, successes=s, success_rate=s / a)
            for name, (a, s) in sorted(per_gesture.items())
        ],
        reviews_due=lesson_repo.count_due(db, utctoday()),
        srs_items=len(lesson_repo.list_srs(db)),
    )


def legacy_export(db: Session) -> dict[str, Any]:
    """Pre-v1 JSON export used by the current Settings page."""
    return {
        "attempts": [
            {
                "gesture_name": a.gesture_id,
                "mode": a.mode,
                "success": a.success,
                "confidence": a.confidence,
                "duration_seconds": a.duration_seconds,
                "created_at": a.created_at.isoformat(),
            }
            for a in lesson_repo.list_attempts(db)
        ],
        "reviews": [
            {
                "gesture_name": s.gesture_id,
                "repetitions": s.repetitions,
                "interval_days": s.interval_days,
                "easiness": s.easiness,
                "due_date": s.due_date.isoformat(),
                "last_quality": s.last_quality,
            }
            for s in lesson_repo.list_srs(db)
        ],
    }


def legacy_import(
    db: Session, attempts: list[tuple[AttemptCreate, Any]], reviews: list[dict[str, Any]]
) -> None:
    """Additive import of the pre-v1 JSON format (attempts + SM-2 states)."""
    for payload, created_at in attempts:
        row = LessonAttempt(**payload.model_dump())
        if created_at is not None:
            row.created_at = created_at
        db.add(row)
    for raw in reviews:
        item = lesson_repo.get_srs(db, raw["gesture_id"])
        if item is None:
            item = lesson_repo.add_srs(
                db,
                SrsItem(
                    gesture_id=raw["gesture_id"],
                    due_date=utctoday(),
                    repetitions=0,
                    interval_days=0,
                    easiness=2.5,
                ),
            )
        item.repetitions, item.interval_days, item.easiness = (
            raw["repetitions"],
            raw["interval_days"],
            raw["easiness"],
        )
        item.due_date = raw["due_date"] or utctoday()
        item.last_quality = raw["last_quality"]
    db.commit()
