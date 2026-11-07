from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from .common import ApiModel


class SettingsOut(ApiModel):
    theme: str
    language: str
    active_model_id: str | None
    active_domain_pack_id: str | None
    confidence_threshold: float | None
    stability_count: int | None
    cooldown_ms: int | None
    speech_enabled: bool
    speech_rate: float
    speech_voice: str | None
    updated_at: datetime


class SettingsUpdate(ApiModel):
    theme: Literal["system", "light", "dark"] | None = None
    language: Literal["ru", "en"] | None = None
    # Empty string clears the active pack.
    active_domain_pack_id: str | None = Field(default=None, max_length=64)
    confidence_threshold: float | None = Field(default=None, ge=0, le=1)
    stability_count: int | None = Field(default=None, ge=1, le=20)
    cooldown_ms: int | None = Field(default=None, ge=0, le=60_000)
    speech_enabled: bool | None = None
    speech_rate: float | None = Field(default=None, ge=0.25, le=4)
    speech_voice: str | None = Field(default=None, max_length=120)
