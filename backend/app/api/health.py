from __future__ import annotations

import platform
import sys

import onnxruntime as ort
from fastapi import APIRouter, Depends

from .. import __version__
from ..dependencies import Container, get_container
from ..schemas.health import HealthResponse, SystemInfo
from ..services.model_service import RuntimeStatus

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health(container: Container = Depends(get_container)) -> HealthResponse:
    database_ok = container.database.ping()
    model_status = container.models.active_status()
    return HealthResponse(
        status="ok" if database_ok and model_status == RuntimeStatus.READY else "degraded",
        database="connected" if database_ok else "unavailable",
        active_model=container.models.active_id,
        model_status=model_status,
        version=__version__,
    )


@router.get("/system/info", response_model=SystemInfo)
def system_info(container: Container = Depends(get_container)) -> SystemInfo:
    database_ok = container.database.ping()
    # No absolute paths: user directories must not leak through the API.
    return SystemInfo(
        backend_version=__version__,
        python_version=sys.version.split()[0],
        os=platform.system(),
        os_release=platform.release(),
        machine=platform.machine(),
        onnxruntime_version=ort.__version__,
        onnx_providers=list(ort.get_available_providers()),
        database="connected" if database_ok else "unavailable",
        database_foreign_keys=container.database.foreign_keys_enabled() if database_ok else False,
        active_model=container.models.active_id,
        model_status=container.models.active_status(),
        hand_detector=container.hands.status,
        websocket_connections=container.connections.count,
    )
