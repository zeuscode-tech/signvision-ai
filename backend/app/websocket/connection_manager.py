"""Tracks open recognition connections and enforces the concurrency limit."""

from __future__ import annotations

import asyncio
from typing import Protocol


class Closable(Protocol):
    connection_id: str

    async def shutdown(self) -> None: ...


class ConnectionManager:
    def __init__(self, max_connections: int) -> None:
        self.max_connections = max_connections
        self._connections: dict[str, Closable] = {}
        self._lock = asyncio.Lock()

    @property
    def count(self) -> int:
        return len(self._connections)

    async def register(self, connection: Closable) -> bool:
        async with self._lock:
            if len(self._connections) >= self.max_connections:
                return False
            self._connections[connection.connection_id] = connection
            return True

    async def unregister(self, connection: Closable) -> None:
        async with self._lock:
            self._connections.pop(connection.connection_id, None)

    async def close_all(self) -> None:
        for connection in list(self._connections.values()):
            await connection.shutdown()
