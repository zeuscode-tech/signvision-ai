from __future__ import annotations

import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import make_settings, start_client


def test_health_reports_ready_model_and_database(client: TestClient) -> None:
    body = client.get("/api/v1/health").json()
    assert body == {
        "status": "ok",
        "database": "connected",
        "active_model": "tiny-sign",
        "model_status": "ready",
        "version": "1.0.0",
    }


def test_missing_model_degrades_honestly_but_other_features_work(workspace: Path) -> None:
    shutil.rmtree(workspace / "models" / "tiny")
    client = start_client(make_settings(workspace))
    try:
        health = client.get("/api/v1/health").json()
        assert health["status"] == "degraded" and health["model_status"] == "model_unavailable"
        assert client.get("/api/v1/lessons").status_code == 200
        assert client.post("/api/v1/attempts", json={"gesture_id": "x", "success": True}).status_code == 201
    finally:
        client.__exit__(None, None, None)


def test_model_without_files_is_reported_unavailable(workspace: Path) -> None:
    (workspace / "models" / "tiny" / "model.onnx").unlink()
    client = start_client(make_settings(workspace))
    try:
        assert client.get("/api/v1/health").json()["model_status"] == "model_unavailable"
        entry = client.get("/api/v1/models/tiny-sign").json()
        assert entry["status"] == "unavailable" and "setup.sh" in entry["status_message"]
    finally:
        client.__exit__(None, None, None)


def test_system_info_has_no_absolute_paths(client: TestClient, workspace: Path) -> None:
    response = client.get("/api/v1/system/info")
    assert response.status_code == 200
    body = response.json()
    assert body["database_foreign_keys"] is True and "CPUExecutionProvider" in body["onnx_providers"]
    assert str(workspace) not in response.text and "/Users/" not in response.text


def test_unified_error_format_and_request_id(client: TestClient) -> None:
    response = client.get(
        "/api/v1/models/missing", headers={"X-Request-ID": "9b2f9d4e-1c1a-4d55-9c2e-1d6f1b3a7a10"}
    )
    error = response.json()["error"]
    assert response.status_code == 404 and error["code"] == "MODEL_NOT_FOUND"
    assert error["request_id"] == "9b2f9d4e-1c1a-4d55-9c2e-1d6f1b3a7a10" == response.headers["x-request-id"]
    invalid = client.post("/api/v1/attempts", json={"gesture_id": "", "success": "maybe"}).json()["error"]
    assert invalid["code"] == "VALIDATION_ERROR" and invalid["details"]
    assert client.get("/api/v1/nope").json()["error"]["code"] == "NOT_FOUND"


def test_unexpected_errors_hide_stack_traces(client: TestClient, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from app.services import lesson_service

    def boom(*_: object) -> None:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(lesson_service, "progress", boom)
    response = client.get("/api/v1/progress")
    assert response.status_code == 500 and response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "secret" not in response.text and "Traceback" not in response.text


def test_non_local_clients_are_rejected(settings) -> None:  # type: ignore[no-untyped-def]
    import asyncio

    import httpx

    from app.main import create_app

    async def call() -> httpx.Response:
        transport = httpx.ASGITransport(app=create_app(settings), client=("10.0.0.7", 5000))
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as remote:
            return await remote.get("/api/v1/health")

    response = asyncio.run(call())
    assert response.status_code == 403 and response.json()["error"]["code"] == "FORBIDDEN"


def test_cors_allows_only_local_frontend(client: TestClient) -> None:
    allowed = client.options(
        "/api/v1/health", headers={"Origin": "http://127.0.0.1:5173", "Access-Control-Request-Method": "GET"}
    )
    denied = client.options(
        "/api/v1/health", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"}
    )
    assert allowed.headers.get("access-control-allow-origin") == "http://127.0.0.1:5173"
    assert "access-control-allow-origin" not in denied.headers


def test_logs_are_rotated_files_with_request_ids(client: TestClient, settings) -> None:  # type: ignore[no-untyped-def]
    import logging

    logging.getLogger("app").setLevel(logging.INFO)
    client.get("/api/v1/health", headers={"X-Request-ID": "11111111-2222-3333-4444-555555555555"})
    for handler in logging.getLogger().handlers:
        handler.flush()
    text = (settings.log_dir / "backend.log").read_text(encoding="utf-8")
    assert "[11111111-2222-3333-4444-555555555555]" in text
