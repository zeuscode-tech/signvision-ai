"""Unified error format: {"error": {code, message, details, request_id}}.

Stack traces are logged server-side and never returned to the client.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .logging import current_request_id

logger = logging.getLogger(__name__)


class ErrorCode(StrEnum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    FORBIDDEN = "FORBIDDEN"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_INVALID = "MODEL_INVALID"
    MODEL_NOT_READY = "MODEL_NOT_READY"
    MODEL_ACTIVE = "MODEL_ACTIVE"
    DOMAIN_PACK_NOT_FOUND = "DOMAIN_PACK_NOT_FOUND"
    DOMAIN_PACK_INVALID = "DOMAIN_PACK_INVALID"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    LESSON_NOT_FOUND = "LESSON_NOT_FOUND"
    INVALID_FRAME = "INVALID_FRAME"
    FRAME_TOO_LARGE = "FRAME_TOO_LARGE"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    DATABASE_ERROR = "DATABASE_ERROR"
    TRAINING_DATA_INSUFFICIENT = "TRAINING_DATA_INSUFFICIENT"
    TRAINING_FAILED = "TRAINING_FAILED"
    IMPORT_FORMAT_INVALID = "IMPORT_FORMAT_INVALID"
    INTERNAL_ERROR = "INTERNAL_ERROR"


DEFAULT_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.VALIDATION_ERROR: "Некорректные данные запроса",
    ErrorCode.NOT_FOUND: "Ресурс не найден",
    ErrorCode.CONFLICT: "Конфликт с текущим состоянием",
    ErrorCode.CONFIRMATION_REQUIRED: "Операция требует явного подтверждения",
    ErrorCode.FORBIDDEN: "Доступ запрещён",
    ErrorCode.MODEL_NOT_FOUND: "Модель распознавания не найдена",
    ErrorCode.MODEL_INVALID: "Модель не прошла проверку",
    ErrorCode.MODEL_NOT_READY: "Модель распознавания не готова",
    ErrorCode.MODEL_ACTIVE: "Нельзя удалить активную модель",
    ErrorCode.DOMAIN_PACK_NOT_FOUND: "Отраслевой словарь не найден",
    ErrorCode.DOMAIN_PACK_INVALID: "Отраслевой словарь повреждён или несовместим с моделью",
    ErrorCode.SESSION_NOT_FOUND: "Сессия не найдена",
    ErrorCode.LESSON_NOT_FOUND: "Урок не найден",
    ErrorCode.INVALID_FRAME: "Некорректный кадр",
    ErrorCode.FRAME_TOO_LARGE: "Кадр превышает допустимый размер",
    ErrorCode.RATE_LIMIT_EXCEEDED: "Превышена допустимая частота кадров",
    ErrorCode.DATABASE_ERROR: "Ошибка локальной базы данных",
    ErrorCode.TRAINING_DATA_INSUFFICIENT: "Недостаточно данных для обучения",
    ErrorCode.TRAINING_FAILED: "Обучение завершилось ошибкой",
    ErrorCode.IMPORT_FORMAT_INVALID: "Некорректный формат файла импорта",
    ErrorCode.INTERNAL_ERROR: "Внутренняя ошибка сервера",
}

DEFAULT_STATUS: dict[ErrorCode, int] = {
    ErrorCode.VALIDATION_ERROR: 422,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.CONFLICT: 409,
    ErrorCode.CONFIRMATION_REQUIRED: 400,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.MODEL_NOT_FOUND: 404,
    ErrorCode.MODEL_INVALID: 422,
    ErrorCode.MODEL_NOT_READY: 409,
    ErrorCode.MODEL_ACTIVE: 409,
    ErrorCode.DOMAIN_PACK_NOT_FOUND: 404,
    ErrorCode.DOMAIN_PACK_INVALID: 422,
    ErrorCode.SESSION_NOT_FOUND: 404,
    ErrorCode.LESSON_NOT_FOUND: 404,
    ErrorCode.INVALID_FRAME: 422,
    ErrorCode.FRAME_TOO_LARGE: 413,
    ErrorCode.RATE_LIMIT_EXCEEDED: 429,
    ErrorCode.DATABASE_ERROR: 500,
    ErrorCode.TRAINING_DATA_INSUFFICIENT: 422,
    ErrorCode.TRAINING_FAILED: 500,
    ErrorCode.IMPORT_FORMAT_INVALID: 422,
    ErrorCode.INTERNAL_ERROR: 500,
}


class AppError(Exception):
    def __init__(
        self,
        code: ErrorCode,
        message: str | None = None,
        *,
        details: Any = None,
        status_code: int | None = None,
    ) -> None:
        self.code = code
        self.message = message or DEFAULT_MESSAGES[code]
        self.details = details
        self.status_code = status_code or DEFAULT_STATUS[code]
        super().__init__(self.message)


def error_body(code: ErrorCode | str, message: str, details: Any = None) -> dict[str, Any]:
    return {
        "error": {
            "code": str(code),
            "message": message,
            "details": details,
            "request_id": current_request_id(),
        }
    }


def _response(status: int, code: ErrorCode | str, message: str, details: Any = None) -> JSONResponse:
    return JSONResponse(status_code=status, content=error_body(code, message, details))


def _validation_details(exc: RequestValidationError) -> list[dict[str, Any]]:
    # Only locations and messages: never echo submitted values (they may be landmarks or frames).
    return [
        {"loc": [str(part) for part in item.get("loc", ())], "msg": str(item.get("msg", ""))}
        for item in exc.errors()
    ]


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("Application error %s: %s", exc.code, exc.message)
        return _response(exc.status_code, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _response(
            422,
            ErrorCode.VALIDATION_ERROR,
            DEFAULT_MESSAGES[ErrorCode.VALIDATION_ERROR],
            _validation_details(exc),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: ErrorCode.NOT_FOUND, 405: ErrorCode.VALIDATION_ERROR, 403: ErrorCode.FORBIDDEN}.get(
            exc.status_code,
            ErrorCode.INTERNAL_ERROR if exc.status_code >= 500 else ErrorCode.VALIDATION_ERROR,
        )
        message = exc.detail if isinstance(exc.detail, str) else DEFAULT_MESSAGES[code]
        return _response(exc.status_code, code, message)

    @app.exception_handler(SQLAlchemyError)
    async def handle_db(_: Request, exc: SQLAlchemyError) -> JSONResponse:
        logger.exception("Database error: %s", type(exc).__name__)
        return _response(500, ErrorCode.DATABASE_ERROR, DEFAULT_MESSAGES[ErrorCode.DATABASE_ERROR])

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", type(exc).__name__)
        return _response(500, ErrorCode.INTERNAL_ERROR, DEFAULT_MESSAGES[ErrorCode.INTERNAL_ERROR])
