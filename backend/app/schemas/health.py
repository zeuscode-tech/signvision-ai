from __future__ import annotations

from typing import Literal

from .common import ApiModel


class HealthResponse(ApiModel):
    status: Literal["ok", "degraded"]
    database: Literal["connected", "unavailable"]
    active_model: str | None
    model_status: str
    version: str


class SystemInfo(ApiModel):
    backend_version: str
    python_version: str
    os: str
    os_release: str
    machine: str
    onnxruntime_version: str
    onnx_providers: list[str]
    database: str
    database_foreign_keys: bool
    active_model: str | None
    model_status: str
    hand_detector: str
    websocket_connections: int
