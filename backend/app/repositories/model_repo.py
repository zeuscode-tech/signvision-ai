from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from ..core.timeutils import utcnow
from ..db_models import ModelRegistryEntry


def get(db: Session, model_id: str) -> ModelRegistryEntry | None:
    return db.get(ModelRegistryEntry, model_id)


def list_models(
    db: Session, *, types: Sequence[str] | None = None, source: str | None = None
) -> list[ModelRegistryEntry]:
    query = select(ModelRegistryEntry).order_by(ModelRegistryEntry.type, ModelRegistryEntry.created_at)
    if types:
        query = query.where(ModelRegistryEntry.type.in_(list(types)))
    if source:
        query = query.where(ModelRegistryEntry.source == source)
    return list(db.scalars(query).all())


def add(db: Session, entry: ModelRegistryEntry) -> ModelRegistryEntry:
    db.add(entry)
    db.flush()
    return entry


def active(db: Session, model_type: str) -> ModelRegistryEntry | None:
    return db.scalar(
        select(ModelRegistryEntry).where(
            ModelRegistryEntry.type == model_type, ModelRegistryEntry.is_active.is_(True)
        )
    )


def set_active(db: Session, entry: ModelRegistryEntry) -> None:
    db.execute(
        update(ModelRegistryEntry)
        .where(ModelRegistryEntry.type == entry.type, ModelRegistryEntry.id != entry.id)
        .values(is_active=False)
    )
    entry.is_active = True
    entry.activated_at = utcnow()
    db.flush()


def deactivate_type(db: Session, model_type: str) -> None:
    db.execute(
        update(ModelRegistryEntry).where(ModelRegistryEntry.type == model_type).values(is_active=False)
    )


def previous_active(db: Session, model_type: str, exclude_id: str) -> ModelRegistryEntry | None:
    """Most recently activated version of this type other than `exclude_id` (for rollback)."""
    return db.scalar(
        select(ModelRegistryEntry)
        .where(
            ModelRegistryEntry.type == model_type,
            ModelRegistryEntry.id != exclude_id,
            ModelRegistryEntry.activated_at.is_not(None),
            ModelRegistryEntry.status == "ready",
        )
        .order_by(ModelRegistryEntry.activated_at.desc())
        .limit(1)
    )


def delete(db: Session, entry: ModelRegistryEntry) -> None:
    db.delete(entry)
    db.flush()
