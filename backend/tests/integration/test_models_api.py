from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import tiny_manifest, write_model_dir


def test_registry_lists_builtin_model_with_status(client: TestClient) -> None:
    models = client.get("/api/v1/models").json()
    tiny = next(m for m in models if m["id"] == "tiny-sign")
    assert tiny["is_active"] and tiny["runtime_status"] == "ready" and tiny["labels_count"] == 4
    detail = client.get("/api/v1/models/tiny-sign").json()
    assert detail["labels"] == ["no", "врач", "голова", "помогать"] and detail["manifest"]["frame_count"] == 4


def test_validate_runs_all_checks(client: TestClient) -> None:
    report = client.post("/api/v1/models/tiny-sign/validate").json()
    assert report["valid"]
    assert {c["name"] for c in report["checks"]} >= {
        "manifest",
        "paths",
        "files",
        "onnx_graph",
        "labels",
        "input_shape",
        "output_size",
        "test_inference",
    }


def test_register_activate_and_delete(client: TestClient, workspace: Path) -> None:
    write_model_dir(workspace / "models" / "second", tiny_manifest("second-sign"))
    created = client.post("/api/v1/models/register", json={"manifest_path": "second/manifest.json"})
    assert created.status_code == 201 and created.json()["status"] == "ready"
    assert (
        client.post("/api/v1/models/register", json={"manifest_path": "second/manifest.json"}).status_code
        == 409
    )
    assert client.post("/api/v1/models/second-sign/activate").json()["is_active"]
    assert client.get("/api/v1/health").json()["active_model"] == "second-sign"
    assert client.delete("/api/v1/models/second-sign").json()["error"]["code"] == "MODEL_ACTIVE"
    client.post("/api/v1/models/tiny-sign/activate")
    assert client.delete("/api/v1/models/second-sign").status_code == 200


def test_label_count_mismatch_is_rejected(client: TestClient, workspace: Path) -> None:
    write_model_dir(workspace / "models" / "bad", tiny_manifest("bad-sign"), n_outputs=6)
    response = client.post("/api/v1/models/register", json={"manifest_path": "bad/manifest.json"})
    error = response.json()["error"]
    assert error["code"] == "MODEL_INVALID"
    assert any(c["name"] == "output_size" and not c["ok"] for c in error["details"]["checks"])


def test_wrong_input_shape_and_corrupted_graph(client: TestClient, workspace: Path) -> None:
    directory = write_model_dir(
        workspace / "models" / "shape",
        tiny_manifest(
            "shape-sign",
            input_shape=[1, 3, 4, 32, 32],
            preprocessing={**tiny_manifest()["preprocessing"], "width": 32, "height": 32},
        ),
    )  # type: ignore[dict-item]
    assert (
        client.post("/api/v1/models/register", json={"manifest_path": "shape/manifest.json"}).json()["error"][
            "code"
        ]
        == "MODEL_INVALID"
    )
    (directory / "model.onnx").write_bytes(b"garbage")
    report = client.post("/api/v1/models/register", json={"manifest_path": "shape/manifest.json"}).json()
    assert report["error"]["code"] == "MODEL_INVALID"


def test_paths_outside_model_root_are_refused(client: TestClient, workspace: Path) -> None:
    outside = write_model_dir(workspace / "outside", tiny_manifest("outside-sign"))
    assert (
        client.post("/api/v1/models/register", json={"manifest_path": "../outside/manifest.json"}).status_code
        == 422
    )
    escape = write_model_dir(
        workspace / "models" / "escape", tiny_manifest("escape-sign", model_path="../../outside/model.onnx")
    )
    error = client.post("/api/v1/models/register", json={"manifest_path": "escape/manifest.json"}).json()[
        "error"
    ]
    assert error["code"] == "MODEL_INVALID" and outside.exists() and escape.exists()


def test_failed_activation_keeps_previous_model(client: TestClient, workspace: Path) -> None:
    directory = write_model_dir(workspace / "models" / "flaky", tiny_manifest("flaky-sign"))
    assert (
        client.post("/api/v1/models/register", json={"manifest_path": "flaky/manifest.json"}).status_code
        == 201
    )
    (directory / "labels.json").write_text(json.dumps(["only-one"]), encoding="utf-8")
    response = client.post("/api/v1/models/flaky-sign/activate")
    assert response.status_code == 422
    health = client.get("/api/v1/health").json()
    assert health["active_model"] == "tiny-sign" and health["model_status"] == "ready"
