"""Recognition sessions, events and aggregated technical metrics."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy.orm import Session

from ..core.exceptions import AppError, ErrorCode
from ..core.timeutils import utcnow
from ..db_models import RecognitionEvent, RecognitionSession
from ..repositories import session_repo

MAX_LATENCY_SAMPLES = 20_000


@dataclass
class SessionMetrics:
    frames_received: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    invalid_frames: int = 0
    predictions: int = 0
    accepted: int = 0
    corrections: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def add_latency(self, value: float) -> None:
        if len(self.latencies_ms) < MAX_LATENCY_SAMPLES:
            self.latencies_ms.append(value)
        else:  # keep memory bounded on very long sessions: overwrite pseudo-randomly
            self.latencies_ms[self.predictions % MAX_LATENCY_SAMPLES] = value

    def latency_summary(self) -> tuple[float | None, float | None, float | None]:
        if not self.latencies_ms:
            return None, None, None
        values = np.asarray(self.latencies_ms, dtype=np.float64)
        return (
            round(float(values.mean()), 2),
            round(float(np.median(values)), 2),
            round(float(np.percentile(values, 95)), 2),
        )


def create(db: Session, *, mode: str, model_id: str | None, domain_pack_id: str | None) -> RecognitionSession:
    session = session_repo.create(
        db,
        RecognitionSession(
            id=str(uuid.uuid4()),
            mode=mode,
            model_id=model_id,
            domain_pack_id=domain_pack_id,
            status="active",
            started_at=utcnow(),
            frames_received=0,
            frames_processed=0,
            frames_dropped=0,
            invalid_frames=0,
            predictions_count=0,
            accepted_count=0,
            corrections_count=0,
        ),
    )
    db.commit()
    return session


def get(db: Session, session_id: str, *, with_events: bool = False) -> RecognitionSession:
    session = session_repo.get(db, session_id, with_events=with_events)
    if session is None:
        raise AppError(ErrorCode.SESSION_NOT_FOUND)
    return session


def list_sessions(db: Session, limit: int, offset: int) -> list[RecognitionSession]:
    return session_repo.list_sessions(db, limit, offset)


def finish(
    db: Session, session_id: str, *, reason: str, metrics: SessionMetrics | None = None
) -> RecognitionSession:
    session = get(db, session_id)
    if metrics is not None:
        session.frames_received += metrics.frames_received
        session.frames_processed += metrics.frames_processed
        session.frames_dropped += metrics.frames_dropped
        session.invalid_frames += metrics.invalid_frames
        session.predictions_count += metrics.predictions
        session.accepted_count += metrics.accepted
        session.corrections_count += metrics.corrections
        avg, median, p95 = metrics.latency_summary()
        if avg is not None:
            session.avg_latency_ms, session.median_latency_ms, session.p95_latency_ms = avg, median, p95
    if session.status != "finished":
        ended = utcnow()
        session.status = "finished"
        session.end_reason = reason
        session.ended_at = ended
        session.duration_ms = int((ended - session.started_at).total_seconds() * 1000)
    db.commit()
    return session


def record_event(
    db: Session,
    session_id: str,
    *,
    state: str,
    predicted_label: str | None,
    confidence: float,
    inference_ms: float,
    accepted: bool,
) -> int | None:
    if session_repo.get(db, session_id) is None:
        return None  # e.g. user data was deleted while the connection was open
    event = session_repo.add_event(
        db,
        RecognitionEvent(
            session_id=session_id,
            state=state,
            predicted_label=predicted_label,
            confidence=confidence,
            inference_ms=round(inference_ms, 2),
            accepted=accepted,
            corrected=False,
            created_at=utcnow(),
        ),
    )
    db.commit()
    return event.id


def correct_event(db: Session, session_id: str, event_id: int, label: str) -> RecognitionEvent:
    event = session_repo.get_event(db, session_id, event_id)
    if event is None:
        raise AppError(ErrorCode.NOT_FOUND, "Событие распознавания не найдено")
    event.selected_label = label
    event.corrected = label != event.predicted_label or not event.accepted
    db.commit()
    return event


def close_interrupted(db: Session) -> int:
    """Sessions left 'active' by a crash or restart are closed at startup."""
    rows = session_repo.active_sessions(db)
    now = utcnow()
    for row in rows:
        row.status = "finished"
        row.end_reason = "interrupted"
        row.ended_at = now
        row.duration_ms = int((now - row.started_at).total_seconds() * 1000)
    db.commit()
    return len(rows)
