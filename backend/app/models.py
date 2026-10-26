from datetime import datetime, date
from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base


class GestureClass(Base):
    __tablename__ = "gesture_classes"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    is_unknown: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    sessions: Mapped[list["CaptureSession"]] = relationship(cascade="all, delete-orphan")


class CaptureSession(Base):
    __tablename__ = "capture_sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    gesture_class_id: Mapped[int] = mapped_column(ForeignKey("gesture_classes.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    note: Mapped[str] = mapped_column(String(200), default="")
    samples: Mapped[list["TrainingSample"]] = relationship(cascade="all, delete-orphan")


class TrainingSample(Base):
    __tablename__ = "training_samples"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("capture_sessions.id"), index=True)
    features_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Attempt(Base):
    __tablename__ = "attempts"
    id: Mapped[int] = mapped_column(primary_key=True)
    gesture_name: Mapped[str] = mapped_column(String(80), index=True)
    mode: Mapped[str] = mapped_column(String(20), default="practice")
    success: Mapped[bool] = mapped_column(Boolean)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ReviewState(Base):
    __tablename__ = "review_states"
    id: Mapped[int] = mapped_column(primary_key=True)
    gesture_name: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    repetitions: Mapped[int] = mapped_column(Integer, default=0)
    interval_days: Mapped[int] = mapped_column(Integer, default=0)
    easiness: Mapped[float] = mapped_column(Float, default=2.5)
    due_date: Mapped[date] = mapped_column(Date, default=date.today)
    last_quality: Mapped[int | None] = mapped_column(Integer, nullable=True)

