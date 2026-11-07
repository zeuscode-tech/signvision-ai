from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from ..db_models import RecognitionEvent, RecognitionSession


def create(db: Session, session: RecognitionSession) -> RecognitionSession:
    db.add(session)
    db.flush()
    return session


def get(db: Session, session_id: str, *, with_events: bool = False) -> RecognitionSession | None:
    if not with_events:
        return db.get(RecognitionSession, session_id)
    return db.scalar(
        select(RecognitionSession)
        .where(RecognitionSession.id == session_id)
        .options(selectinload(RecognitionSession.events))
    )


def list_sessions(db: Session, limit: int, offset: int) -> list[RecognitionSession]:
    return list(
        db.scalars(
            select(RecognitionSession)
            .order_by(RecognitionSession.started_at.desc())
            .limit(limit)
            .offset(offset)
        ).all()
    )


def add_event(db: Session, event: RecognitionEvent) -> RecognitionEvent:
    db.add(event)
    db.flush()
    return event


def get_event(db: Session, session_id: str, event_id: int) -> RecognitionEvent | None:
    event = db.get(RecognitionEvent, event_id)
    return event if event is not None and event.session_id == session_id else None


def active_sessions(db: Session) -> list[RecognitionSession]:
    return list(db.scalars(select(RecognitionSession).where(RecognitionSession.status == "active")).all())


def delete_all(db: Session) -> int:
    count = db.scalar(select(func.count(RecognitionSession.id))) or 0
    db.execute(delete(RecognitionEvent))
    db.execute(delete(RecognitionSession))
    return int(count)
