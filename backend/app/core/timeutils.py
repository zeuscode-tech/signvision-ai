"""All timestamps are timezone-aware UTC."""

from __future__ import annotations

from datetime import UTC, date, datetime


def utcnow() -> datetime:
    return datetime.now(UTC)


def utctoday() -> date:
    return utcnow().date()


def isoformat_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
