from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..core.timeutils import utcnow
from ..database.base import Base, UTCDateTime


class RecognitionSession(Base):
    """Aggregated technical metrics of one recognition session. No frames, no images."""

    __tablename__ = "recognition_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    mode: Mapped[str] = mapped_column(String(32))
    model_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    domain_pack_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="active", index=True)
    end_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, index=True)
    ended_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frames_received: Mapped[int] = mapped_column(Integer, default=0)
    frames_processed: Mapped[int] = mapped_column(Integer, default=0)
    frames_dropped: Mapped[int] = mapped_column(Integer, default=0)
    invalid_frames: Mapped[int] = mapped_column(Integer, default=0)
    predictions_count: Mapped[int] = mapped_column(Integer, default=0)
    accepted_count: Mapped[int] = mapped_column(Integer, default=0)
    corrections_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    median_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    p95_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    events: Mapped[list[RecognitionEvent]] = relationship(
        back_populates="session", cascade="all, delete-orphan", passive_deletes=True
    )


class RecognitionEvent(Base):
    """One inference result. Images are never stored."""

    __tablename__ = "recognition_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("recognition_sessions.id", ondelete="CASCADE"), index=True
    )
    state: Mapped[str] = mapped_column(String(16))
    predicted_label: Mapped[str | None] = mapped_column(String(120), nullable=True)
    selected_label: Mapped[str | None] = mapped_column(String(120), nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    inference_ms: Mapped[float] = mapped_column(Float)
    accepted: Mapped[bool] = mapped_column(Boolean, default=False)
    corrected: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)

    session: Mapped[RecognitionSession] = relationship(back_populates="events")
