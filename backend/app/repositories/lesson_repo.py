from __future__ import annotations

from datetime import date

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from ..db_models import Lesson, LessonAttempt, SrsItem


def list_lessons(db: Session) -> list[Lesson]:
    return list(db.scalars(select(Lesson).order_by(Lesson.level, Lesson.id)).all())


def get_lesson(db: Session, lesson_id: str) -> Lesson | None:
    return db.get(Lesson, lesson_id)


def add_lesson(db: Session, lesson: Lesson) -> None:
    db.add(lesson)


def add_attempt(db: Session, attempt: LessonAttempt) -> LessonAttempt:
    db.add(attempt)
    db.flush()
    return attempt


def list_attempts(db: Session) -> list[LessonAttempt]:
    return list(db.scalars(select(LessonAttempt).order_by(LessonAttempt.created_at)).all())


def get_srs(db: Session, gesture_id: str) -> SrsItem | None:
    return db.scalar(select(SrsItem).where(SrsItem.gesture_id == gesture_id))


def add_srs(db: Session, item: SrsItem) -> SrsItem:
    db.add(item)
    db.flush()
    return item


def list_srs(db: Session) -> list[SrsItem]:
    return list(db.scalars(select(SrsItem).order_by(SrsItem.due_date, SrsItem.gesture_id)).all())


def list_due(db: Session, today: date, limit: int) -> list[SrsItem]:
    return list(
        db.scalars(
            select(SrsItem)
            .where(SrsItem.due_date <= today)
            .order_by(SrsItem.due_date, SrsItem.easiness, SrsItem.gesture_id)
            .limit(limit)
        ).all()
    )


def count_due(db: Session, today: date) -> int:
    return int(db.scalar(select(func.count(SrsItem.id)).where(SrsItem.due_date <= today)) or 0)


def delete_progress(db: Session) -> tuple[int, int]:
    attempts = int(db.scalar(select(func.count(LessonAttempt.id))) or 0)
    items = int(db.scalar(select(func.count(SrsItem.id))) or 0)
    db.execute(delete(LessonAttempt))
    db.execute(delete(SrsItem))
    return attempts, items
