from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..core.timeutils import utcnow
from ..database.base import Base, UTCDateTime


class ModelRegistryEntry(Base):
    __tablename__ = "model_registry"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    version: Mapped[str] = mapped_column(String(40))
    # isolated_sign (ONNX video model) | static_landmarks | sequence_landmarks (Training Studio)
    type: Mapped[str] = mapped_column(String(32), index=True)
    # builtin | registered | user
    source: Mapped[str] = mapped_column(String(16), default="registered")
    # Paths are relative to MODEL_ROOT; absolute user paths are never stored or returned.
    manifest_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    model_path: Mapped[str] = mapped_column(String(500))
    labels_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    labels_count: Mapped[int] = mapped_column(Integer, default=0)
    preprocessing_json: Mapped[str] = mapped_column(Text, default="{}")
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # unknown | ready | invalid | unavailable
    status: Mapped[str] = mapped_column(String(16), default="unknown")
    status_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    activated_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow, onupdate=utcnow)
