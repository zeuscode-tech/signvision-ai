"""Logging with rotation and a request id on every record.

Only technical events are logged: startup, database state, model loading,
sessions, inference timings, errors, training and data operations. Frames,
landmarks, training file contents and recognised text are never logged.
"""

from __future__ import annotations

import logging
import logging.handlers
import uuid
from contextvars import ContextVar
from pathlib import Path

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

LOG_FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"


def new_request_id() -> str:
    return str(uuid.uuid4())


def current_request_id() -> str:
    return request_id_var.get()


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


_configured_handlers: list[logging.Handler] = []


def configure_logging(log_dir: Path, level: str = "INFO") -> None:
    root = logging.getLogger()
    for handler in _configured_handlers:
        root.removeHandler(handler)
        handler.close()
    _configured_handlers.clear()

    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT)
    request_filter = RequestIdFilter()

    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "backend.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    console_handler = logging.StreamHandler()
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        handler.addFilter(request_filter)
        root.addHandler(handler)
        _configured_handlers.append(handler)
    root.setLevel(level)
    # Third-party libraries are noisy at INFO and some log request details.
    for noisy in ("uvicorn.access", "multipart", "PIL", "matplotlib", "absl"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
