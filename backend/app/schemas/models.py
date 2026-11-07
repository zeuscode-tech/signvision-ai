from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from .common import ApiModel


class ModelOut(ApiModel):
    id: str
    name: str
    version: str
    type: str
    source: str
    manifest_path: str | None
    model_path: str
    labels_path: str | None
    labels_count: int
    preprocessing: dict[str, Any]
    metrics: dict[str, Any] | None
    status: str
    status_message: str | None
    runtime_status: str
    is_active: bool
    activated_at: datetime | None
    created_at: datetime


class ModelDetail(ModelOut):
    labels: list[str] | None = None
    manifest: dict[str, Any] | None = None


class RegisterModelIn(ApiModel):
    # Path of the manifest relative to MODEL_ROOT, e.g. "my-model/manifest.json".
    manifest_path: str = Field(min_length=1, max_length=300)


class ValidationCheckOut(ApiModel):
    name: str
    ok: bool
    message: str


class ValidationReportOut(ApiModel):
    model_id: str | None
    valid: bool
    checks: list[ValidationCheckOut]
