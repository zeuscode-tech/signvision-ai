from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..core.timeutils import utcnow
from ..database.base import Base, UTCDateTime


class TrainingClass(Base):
    __tablename__ = "training_classes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    # casefold()-ed name: SQLite lower() does not handle Cyrillic.
    normalized_name: Mapped[str] = mapped_column(String(80), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    is_negative: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    sessions: Mapped[list[TrainingSession]] = relationship(
        back_populates="training_class", cascade="all, delete-orphan", passive_deletes=True
    )


class TrainingSession(Base):
    """One independent capture session. Samples of a session never span train and test."""

    __tablename__ = "training_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    class_id: Mapped[int] = mapped_column(ForeignKey("training_classes.id", ondelete="CASCADE"), index=True)
    note: Mapped[str] = mapped_column(String(200), default="")
    image_consent: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    training_class: Mapped[TrainingClass] = relationship(back_populates="sessions")
    samples: Mapped[list[TrainingSample]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )


class TrainingSample(Base):
    __tablename__ = "training_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("training_sessions.id", ondelete="CASCADE"), index=True
    )
    # static | sequence
    kind: Mapped[str] = mapped_column(String(16), default="static", index=True)
    features_json: Mapped[str] = mapped_column(Text)
    frame_count: Mapped[int] = mapped_column(Integer, default=1)
    # Relative to DATA_DIR; only set when the session has explicit image consent.
    image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    session: Mapped[TrainingSession] = relationship(back_populates="samples")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    # queued | running | completed | failed
    status: Mapped[str] = mapped_column(String(16), index=True)
    sample_kind: Mapped[str] = mapped_column(String(16), default="static")
    activate_on_success: Mapped[bool] = mapped_column(Boolean, default=False)
    dataset_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_registry.id", ondelete="SET NULL"), nullable=True
    )
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
