from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import case, delete, func, select
from sqlalchemy.orm import Session

from ..db_models import TrainingClass, TrainingJob, TrainingSample, TrainingSession


@dataclass
class ClassCounts:
    sessions: int = 0
    samples: int = 0
    static: int = 0
    sequence: int = 0


def list_classes(db: Session) -> list[TrainingClass]:
    return list(db.scalars(select(TrainingClass).order_by(TrainingClass.created_at, TrainingClass.id)).all())


def get_class(db: Session, class_id: int) -> TrainingClass | None:
    return db.get(TrainingClass, class_id)


def find_class_by_name(db: Session, normalized_name: str) -> TrainingClass | None:
    return db.scalar(select(TrainingClass).where(TrainingClass.normalized_name == normalized_name))


def add(db: Session, row: TrainingClass | TrainingSession | TrainingSample | TrainingJob) -> None:
    db.add(row)
    db.flush()


def delete_row(db: Session, row: TrainingClass | TrainingSession | TrainingSample) -> None:
    db.delete(row)
    db.flush()


def class_counts(db: Session) -> dict[int, ClassCounts]:
    counts: dict[int, ClassCounts] = {}
    for class_id, sessions in db.execute(
        select(TrainingSession.class_id, func.count(TrainingSession.id)).group_by(TrainingSession.class_id)
    ):
        counts.setdefault(class_id, ClassCounts()).sessions = int(sessions)
    rows = db.execute(
        select(
            TrainingSession.class_id,
            func.count(TrainingSample.id),
            func.sum(case((TrainingSample.kind == "static", 1), else_=0)),
            func.sum(case((TrainingSample.kind == "sequence", 1), else_=0)),
        )
        .join(TrainingSample, TrainingSample.session_id == TrainingSession.id)
        .group_by(TrainingSession.class_id)
    )
    for class_id, total, static, sequence in rows:
        item = counts.setdefault(class_id, ClassCounts())
        item.samples, item.static, item.sequence = int(total), int(static or 0), int(sequence or 0)
    return counts


def get_session(db: Session, session_id: int) -> TrainingSession | None:
    return db.get(TrainingSession, session_id)


def list_sessions(db: Session, class_id: int | None = None) -> list[TrainingSession]:
    query = select(TrainingSession).order_by(TrainingSession.id)
    if class_id is not None:
        query = query.where(TrainingSession.class_id == class_id)
    return list(db.scalars(query).all())


def session_sample_counts(db: Session) -> dict[int, int]:
    return {
        sid: int(n)
        for sid, n in db.execute(
            select(TrainingSample.session_id, func.count(TrainingSample.id)).group_by(
                TrainingSample.session_id
            )
        )
    }


def get_sample(db: Session, sample_id: int) -> TrainingSample | None:
    return db.get(TrainingSample, sample_id)


def list_samples(
    db: Session, *, session_id: int | None, class_id: int | None, kind: str | None, limit: int, offset: int
) -> list[tuple[TrainingSample, int]]:
    query = (
        select(TrainingSample, TrainingSession.class_id)
        .join(TrainingSession, TrainingSample.session_id == TrainingSession.id)
        .order_by(TrainingSample.id.desc())
        .limit(limit)
        .offset(offset)
    )
    if session_id is not None:
        query = query.where(TrainingSample.session_id == session_id)
    if class_id is not None:
        query = query.where(TrainingSession.class_id == class_id)
    if kind is not None:
        query = query.where(TrainingSample.kind == kind)
    return [(sample, int(cid)) for sample, cid in db.execute(query).all()]


def dataset_rows(db: Session, kind: str) -> list[tuple[TrainingSample, TrainingSession, TrainingClass]]:
    query = (
        select(TrainingSample, TrainingSession, TrainingClass)
        .join(TrainingSession, TrainingSample.session_id == TrainingSession.id)
        .join(TrainingClass, TrainingSession.class_id == TrainingClass.id)
        .where(TrainingSample.kind == kind)
        .order_by(TrainingSample.id)
    )
    return [(a, b, c) for a, b, c in db.execute(query).all()]


def all_samples(db: Session) -> list[TrainingSample]:
    return list(db.scalars(select(TrainingSample).order_by(TrainingSample.id)).all())


def get_job(db: Session, job_id: str) -> TrainingJob | None:
    return db.get(TrainingJob, job_id)


def unfinished_jobs(db: Session) -> list[TrainingJob]:
    return list(db.scalars(select(TrainingJob).where(TrainingJob.status.in_(["queued", "running"]))).all())


def image_paths(db: Session) -> list[str]:
    return [
        p
        for p in db.scalars(select(TrainingSample.image_path).where(TrainingSample.image_path.is_not(None)))
        if p
    ]


def delete_all(db: Session) -> dict[str, int]:
    counts = {
        "training_classes": int(db.scalar(select(func.count(TrainingClass.id))) or 0),
        "training_sessions": int(db.scalar(select(func.count(TrainingSession.id))) or 0),
        "training_samples": int(db.scalar(select(func.count(TrainingSample.id))) or 0),
        "training_jobs": int(db.scalar(select(func.count(TrainingJob.id))) or 0),
    }
    db.execute(delete(TrainingJob))
    db.execute(delete(TrainingSample))
    db.execute(delete(TrainingSession))
    db.execute(delete(TrainingClass))
    return counts
