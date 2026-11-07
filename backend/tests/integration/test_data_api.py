from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import make_settings, start_client
from tests.integration.test_training_api import build_dataset, wait_job

CONFIRM = {"confirm": "DELETE_ALL_USER_DATA"}


def seed(client: TestClient) -> str:
    build_dataset(client, sessions=2, samples=3)
    client.post("/api/v1/attempts", json={"gesture_id": "Мой жест", "success": True, "lesson_id": "basics"})
    client.put("/api/v1/settings", json={"theme": "dark", "confidence_threshold": 0.7})
    job = wait_job(
        client, client.post("/api/v1/training/jobs", json={"activate_on_success": True}).json()["id"]
    )
    return str(job["model_id"])


def export(client: TestClient, include_models: bool = False) -> bytes:
    response = client.get("/api/v1/data/export", params={"include_models": include_models})
    assert response.status_code == 200 and response.headers["content-type"] == "application/zip"
    return response.content


def upload(client: TestClient, data: bytes):  # type: ignore[no-untyped-def]
    return client.post("/api/v1/data/import", files={"file": ("export.zip", data, "application/zip")})


def rewrite(data: bytes, changes: dict[str, bytes | None], extra: dict[str, bytes] | None = None) -> bytes:
    source = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as target:
        for info in source.infolist():
            if info.filename in changes:
                if changes[info.filename] is not None:
                    target.writestr(info.filename, changes[info.filename])  # type: ignore[arg-type]
            else:
                target.writestr(info, source.read(info.filename))
        for name, content in (extra or {}).items():
            target.writestr(name, content)
    return out.getvalue()


def counts(client: TestClient) -> tuple[int, int, int]:
    classes = client.get("/api/v1/training/classes").json()
    return (
        len(classes),
        sum(c["sample_count"] for c in classes),
        client.get("/api/v1/progress").json()["attempts"],
    )


def test_export_contains_versioned_manifest_and_no_images(client: TestClient) -> None:
    seed(client)
    archive = zipfile.ZipFile(io.BytesIO(export(client)))
    names = set(archive.namelist())
    assert names == {"manifest.json", "settings.json", "progress.json", "training.json"}
    manifest = json.loads(archive.read("manifest.json"))
    assert manifest["format"] == "signvision-user-data" and manifest["format_version"] == 1
    assert set(manifest["files"]) == names - {"manifest.json"}
    with_models = set(zipfile.ZipFile(io.BytesIO(export(client, include_models=True))).namelist())
    assert "models/user-static-v1/model.joblib" in with_models


def test_import_roundtrip_replaces_data(client: TestClient) -> None:
    seed(client)
    before = counts(client)
    data = export(client, include_models=True)
    client.delete("/api/v1/data", params=CONFIRM)
    assert counts(client) == (0, 0, 0)
    result = upload(client, data).json()
    assert result["imported_samples"] == before[1] and result["imported_models"] == ["user-static-v1"]
    assert counts(client) == before
    assert client.get("/api/v1/settings").json()["theme"] == "dark"
    assert client.post("/api/v1/training/models/user-static-v1/activate").status_code == 200


def test_models_from_another_installation_are_skipped(client: TestClient, workspace: Path) -> None:
    seed(client)
    data = export(client, include_models=True)
    other = start_client(make_settings(workspace / "other"))
    try:
        result = upload(other, data).json()
        assert result["imported_models"] == []
        assert result["skipped_models"] == [{"id": "user-static-v1", "reason": "signature_mismatch"}]
        assert result["imported_samples"] > 0
    finally:
        other.__exit__(None, None, None)


def test_tampered_archive_is_rejected_without_changing_data(client: TestClient) -> None:
    seed(client)
    before = counts(client)
    data = export(client)
    training = json.loads(zipfile.ZipFile(io.BytesIO(data)).read("training.json"))
    training["classes"][0]["name"] = "подмена"
    response = upload(client, rewrite(data, {"training.json": json.dumps(training).encode()}))
    assert response.status_code == 422 and response.json()["error"]["code"] == "IMPORT_FORMAT_INVALID"
    assert counts(client) == before


def test_path_traversal_symlinks_and_unknown_members_are_rejected(
    client: TestClient, workspace: Path
) -> None:
    data = export(client)
    for extra in ({"../evil.txt": b"x"}, {"/abs.txt": b"x"}, {"notes.txt": b"x"}, {"models/x/run.py": b"x"}):
        response = upload(client, rewrite(data, {}, extra))
        assert response.status_code == 422, extra
    assert not (workspace / "evil.txt").exists()
    link = io.BytesIO()
    with zipfile.ZipFile(link, "w") as archive:
        info = zipfile.ZipInfo("settings.json")
        info.external_attr = 0o120777 << 16
        archive.writestr(info, "/etc/passwd")
    assert upload(client, link.getvalue()).status_code == 422
    assert upload(client, b"not a zip").json()["error"]["code"] == "IMPORT_FORMAT_INVALID"


def test_unsupported_version_and_size_limit(workspace: Path) -> None:
    client = start_client(make_settings(workspace, max_import_size_bytes=2_000))
    try:
        assert upload(client, b"0" * 3_000).json()["error"]["message"].startswith("Файл импорта превышает")
    finally:
        client.__exit__(None, None, None)
    client = start_client(make_settings(workspace))
    try:
        data = export(client)
        manifest = json.loads(zipfile.ZipFile(io.BytesIO(data)).read("manifest.json"))
        manifest["format_version"] = 99
        response = upload(client, rewrite(data, {"manifest.json": json.dumps(manifest).encode()}))
        assert "версия" in response.json()["error"]["message"]
    finally:
        client.__exit__(None, None, None)


def test_delete_requires_confirmation_and_keeps_builtin_models(client: TestClient, settings) -> None:  # type: ignore[no-untyped-def]
    model_id = seed(client)
    assert client.delete("/api/v1/data").json()["error"]["code"] == "CONFIRMATION_REQUIRED"
    result = client.delete("/api/v1/data", params=CONFIRM).json()
    assert (
        result["deleted"]
        and result["removed"]["training_samples"] == 12
        and result["removed"]["user_models"] == 1
    )
    assert counts(client) == (0, 0, 0)
    assert not (settings.user_model_root / model_id).exists()
    assert client.get("/api/v1/health").json()["model_status"] == "ready"
    assert (settings.model_root / "tiny" / "model.onnx").exists()
