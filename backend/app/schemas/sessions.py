from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from .common import ApiModel


class SessionCreate(ApiModel):
    mode: Literal["isolated_sign"] = "isolated_sign"
    model_id: str | None = Field(default=None, max_length=64)
    domain_pack_id: str | None = Field(default=None, max_length=64)


class SessionOut(ApiModel):
    id: str
    mode: str
    model_id: str | None
    domain_pack_id: str | None
    status: str
    end_reason: str | None
    started_at: datetime
    ended_at: datetime | None
    duration_ms: int | None
    frames_received: int
    frames_processed: int
    frames_dropped: int
    invalid_frames: int
    predictions_count: int
    accepted_count: int
    corrections_count: int
    avg_latency_ms: float | None
    median_latency_ms: float | None
    p95_latency_ms: float | None


class EventOut(ApiModel):
    id: int
    state: str
    predicted_label: str | None
    selected_label: str | None
    confidence: float
    inference_ms: float
    accepted: bool
    corrected: bool
    created_at: datetime


class SessionDetail(SessionOut):
    events: list[EventOut]
