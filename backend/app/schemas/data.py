from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EXPORT_FORMAT = "signvision-user-data"
EXPORT_FORMAT_VERSION = 1


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ExportManifest(Strict):
    format: Literal["signvision-user-data"]
    format_version: int
    app_version: str
    created_at: datetime
    includes_models: bool
    files: dict[str, str]  # member name -> sha256


class SettingsRecord(Strict):
    theme: str = Field(max_length=16)
    language: str = Field(max_length=16)
    active_domain_pack_id: str | None = Field(default=None, max_length=64)
    confidence_threshold: float | None = Field(default=None, ge=0, le=1)
    stability_count: int | None = Field(default=None, ge=1, le=20)
    cooldown_ms: int | None = Field(default=None, ge=0, le=60_000)
    speech_enabled: bool = False
    speech_rate: float = Field(default=1.0, ge=0.25, le=4)
    speech_voice: str | None = Field(default=None, max_length=120)


class AttemptRecord(Strict):
    lesson_id: str | None = Field(default=None, max_length=64)
    gesture_id: str = Field(min_length=1, max_length=120)
    mode: str = Field(max_length=20)
    success: bool
    confidence: float | None = Field(default=None, ge=0, le=1)
    duration_seconds: int = Field(ge=0, le=86_400)
    created_at: datetime


class SrsRecord(Strict):
    gesture_id: str = Field(min_length=1, max_length=120)
    repetitions: int = Field(ge=0, le=10_000)
    interval_days: int = Field(ge=0, le=36_500)
    easiness: float = Field(ge=1.3, le=5)
    due_date: date
    last_quality: int | None = Field(default=None, ge=0, le=5)
    last_reviewed_at: datetime | None = None


class ProgressFile(Strict):
    lesson_attempts: list[AttemptRecord] = Field(max_length=500_000)
    srs_items: list[SrsRecord] = Field(max_length=100_000)


class ClassRecord(Strict):
    id: int
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(max_length=2000)
    is_negative: bool
    created_at: datetime


class SessionRecord(Strict):
    id: int
    class_id: int
    note: str = Field(max_length=200)
    image_consent: bool
    created_at: datetime


class SampleRecord(Strict):
    id: int
    session_id: int
    kind: Literal["static", "sequence"]
    features: list[float] = Field(max_length=4096)
    frame_count: int = Field(ge=1, le=120)
    created_at: datetime


class UserModelRecord(Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{1,63}$")
    name: str = Field(max_length=200)
    version: str = Field(max_length=40)
    type: Literal["static_landmarks", "sequence_landmarks"]
    labels_count: int = Field(ge=0)
    metrics: dict[str, Any] | None
    preprocessing: dict[str, Any]
    created_at: datetime
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")


class TrainingFile(Strict):
    classes: list[ClassRecord] = Field(max_length=10_000)
    sessions: list[SessionRecord] = Field(max_length=100_000)
    samples: list[SampleRecord] = Field(max_length=1_000_000)
    models: list[UserModelRecord] = Field(default_factory=list, max_length=1_000)


class ImportResult(BaseModel):
    imported_attempts: int
    imported_srs_items: int
    imported_classes: int
    imported_sessions: int
    imported_samples: int
    imported_models: list[str]
    skipped_models: list[dict[str, str]]


class DeleteResult(BaseModel):
    deleted: bool
    removed: dict[str, int]
