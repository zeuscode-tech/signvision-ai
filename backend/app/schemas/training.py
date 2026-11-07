from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from .common import ApiModel

SampleKind = Literal["static", "sequence"]


class TrainingClassCreate(ApiModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=2000)
    # Negative / "unknown" class: rejected at prediction time instead of being returned as a word.
    is_negative: bool = False

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Название не может быть пустым")
        return value


class TrainingClassOut(ApiModel):
    id: int
    name: str
    description: str
    is_negative: bool
    session_count: int
    sample_count: int
    static_samples: int
    sequence_samples: int
    created_at: datetime


class TrainingSessionCreate(ApiModel):
    class_id: int
    note: str = Field(default="", max_length=200)
    # Images are stored only with explicit consent for this session.
    image_consent: bool = False


class TrainingSessionOut(ApiModel):
    id: int
    class_id: int
    note: str
    image_consent: bool
    sample_count: int
    created_at: datetime


class SampleCreate(ApiModel):
    session_id: int
    kind: SampleKind = "static"
    # static: 126 values = Left hand 21×xyz then Right hand 21×xyz (missing hand = zeros)
    landmarks: list[float] | None = Field(default=None, max_length=126)
    # sequence: 4..120 frames of 126 values each
    sequence: list[list[float]] | None = Field(default=None, max_length=120)
    # Optional JPEG (base64 or data URL); requires image_consent on the session.
    image: str | None = Field(default=None, max_length=3_000_000)

    @model_validator(mode="after")
    def payload_matches_kind(self) -> SampleCreate:
        if self.kind == "static" and self.landmarks is None:
            raise ValueError("Для kind=static нужен landmarks")
        if self.kind == "sequence" and self.sequence is None:
            raise ValueError("Для kind=sequence нужен sequence")
        return self


class SampleOut(ApiModel):
    id: int
    session_id: int
    class_id: int
    kind: str
    frame_count: int
    has_image: bool
    created_at: datetime


class TrainingJobCreate(ApiModel):
    sample_kind: SampleKind = "static"
    activate_on_success: bool = False


class TrainingJobOut(ApiModel):
    id: str
    status: Literal["queued", "running", "completed", "failed"]
    sample_kind: str
    activate_on_success: bool
    dataset_version: str | None
    model_id: str | None
    metrics: dict[str, Any] | None
    error_code: str | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class TrainingModelOut(ApiModel):
    id: str
    name: str
    version: str
    type: str
    labels_count: int
    metrics: dict[str, Any] | None
    is_active: bool
    activated_at: datetime | None
    created_at: datetime


class PredictIn(ApiModel):
    kind: SampleKind = "static"
    landmarks: list[float] | None = Field(default=None, max_length=126)
    sequence: list[list[float]] | None = Field(default=None, max_length=120)


class PredictOut(ApiModel):
    state: Literal["recognized", "uncertain", "unknown"]
    label: str | None
    candidate: str
    confidence: float
    accepted: bool
    is_negative: bool
    top3: list[dict[str, Any]]
    model_id: str
