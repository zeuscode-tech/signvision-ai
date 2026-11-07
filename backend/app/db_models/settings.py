from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from ..core.timeutils import utcnow
from ..database.base import Base, UTCDateTime


class AppSettings(Base):
    """Single-row table (id = 1) with user preferences."""

    __tablename__ = "settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    theme: Mapped[str] = mapped_column(String(16), default="system")
    language: Mapped[str] = mapped_column(String(16), default="ru")
    active_model_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    active_domain_pack_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    stability_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cooldown_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    speech_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    speech_rate: Mapped[float] = mapped_column(Float, default=1.0)
    speech_voice: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)
