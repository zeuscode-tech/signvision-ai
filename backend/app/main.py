"""SignBridge backend entry point: `uvicorn app.main:app --host 127.0.0.1 --port 8000`."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .api import data, domain_packs, health, legacy, lessons, models, progress, sessions, training
from .api import settings as settings_api
from .core.config import Settings, ensure_directories, get_settings
from .core.exceptions import ErrorCode, error_body, register_exception_handlers
from .core.logging import configure_logging, request_id_var
from .core.security import is_loopback_client
from .database.migrations import upgrade_to_head
from .dependencies import Container
from .services import lesson_service, session_service
from .websocket import recognition

logger = logging.getLogger("app")
API_PREFIX = "/api/v1"


def _request_id(header: str | None) -> str:
    if header:
        try:
            return str(uuid.UUID(header))
        except ValueError:
            pass
    return str(uuid.uuid4())


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        ensure_directories(settings)
        configure_logging(settings.log_dir, settings.log_level)
        logger.info(
            "SignBridge backend %s starting on %s:%s", __version__, settings.app_host, settings.app_port
        )
        container = Container.build(settings)
        app.state.container = container
        upgrade_to_head(container.database.engine)
        with container.database.session() as db:
            lesson_service.seed_lessons(db)
            closed = session_service.close_interrupted(db)
            failed = container.training.mark_interrupted(db)
        if closed or failed:
            logger.info("Closed %d interrupted sessions, %d interrupted training jobs", closed, failed)
        logger.info("Database connected (foreign keys: %s)", container.database.foreign_keys_enabled())
        container.models.discover_builtin()
        container.models.start_background_load()  # missing model must not block the server
        logger.info("Hand detector: %s", container.hands.status)
        try:
            yield
        finally:
            await container.connections.close_all()
            container.shutdown()
            logger.info("SignBridge backend stopped")

    app = FastAPI(
        title="SignBridge Backend",
        version=__version__,
        lifespan=lifespan,
        description="Локальный backend SignVision AI / SignBridge Live. Данные не покидают устройство.",
    )
    register_exception_handlers(app)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = _request_id(request.headers.get("x-request-id"))
        request_id_var.set(request_id)
        client = request.client.host if request.client else None
        if not is_loopback_client(client):
            return JSONResponse(
                status_code=403,
                content=error_body(ErrorCode.FORBIDDEN, "Сервис принимает только локальные подключения"),
            )
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "%s %s -> %d (%.0f ms)",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )

    for router in (
        health.router,
        settings_api.router,
        models.router,
        sessions.router,
        domain_packs.router,
        lessons.router,
        progress.router,
        training.router,
        data.router,
    ):
        app.include_router(router, prefix=API_PREFIX)
    app.include_router(recognition.router)
    app.include_router(legacy.router)
    return app


app = create_app()
