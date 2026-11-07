from __future__ import annotations

import base64

import numpy as np
from fastapi.testclient import TestClient

from tests.conftest import RED, jpeg

RNG = np.random.default_rng(7)


def landmarks(shape: str, noise: float = 0.003) -> list[float]:
    """Synthetic right hand: 'line' = fingers along x, 'fan' = fingers along y."""
    points = []
    for i in range(21):
        x, y = (0.3 + i * 0.01, 0.5) if shape == "line" else (0.3, 0.5 - i * 0.01)
        points.extend([x + RNG.normal(0, noise), y + RNG.normal(0, noise), 0.0])
    return [0.0] * 63 + points


def wait_job(client: TestClient, job_id: str) -> dict:  # type: ignore[type-arg]
    client.app.state.container.training.wait_for_job(job_id, timeout=60)  # type: ignore[attr-defined]
    return client.get(f"/api/v1/training/jobs/{job_id}").json()  # type: ignore[no-any-return]


def build_dataset(client: TestClient, sessions: int = 2, samples: int = 4) -> dict[str, int]:
    ids = {}
    for name, shape, negative in (("Мой жест", "line", False), ("неизвестно", "fan", True)):
        class_id = client.post(
            "/api/v1/training/classes", json={"name": name, "is_negative": negative}
        ).json()["id"]
        ids[name] = class_id
        for _ in range(sessions):
            session_id = client.post("/api/v1/training/sessions", json={"class_id": class_id}).json()["id"]
            for _ in range(samples):
                response = client.post(
                    "/api/v1/training/samples", json={"session_id": session_id, "landmarks": landmarks(shape)}
                )
                assert response.status_code == 201
    return ids


def test_classes_sessions_and_samples(client: TestClient) -> None:
    ids = build_dataset(client, sessions=1, samples=2)
    classes = {c["name"]: c for c in client.get("/api/v1/training/classes").json()}
    assert classes["Мой жест"]["session_count"] == 1 and classes["Мой жест"]["sample_count"] == 2
    assert client.post("/api/v1/training/classes", json={"name": "мой ЖЕСТ"}).status_code == 409
    samples = client.get("/api/v1/training/samples", params={"class_id": ids["Мой жест"]}).json()
    assert len(samples) == 2 and "features" not in samples[0]
    assert client.delete(f"/api/v1/training/samples/{samples[0]['id']}").status_code == 200
    assert client.delete(f"/api/v1/training/classes/{ids['неизвестно']}").status_code == 200
    assert len(client.get("/api/v1/training/samples").json()) == 1


def test_samples_without_hands_are_rejected(client: TestClient) -> None:
    class_id = client.post("/api/v1/training/classes", json={"name": "x"}).json()["id"]
    session_id = client.post("/api/v1/training/sessions", json={"class_id": class_id}).json()["id"]
    response = client.post(
        "/api/v1/training/samples", json={"session_id": session_id, "landmarks": [0.0] * 126}
    )
    assert response.status_code == 422 and "Руки" in response.json()["error"]["message"]


def test_insufficient_independent_sessions_are_refused(client: TestClient) -> None:
    build_dataset(client, sessions=1, samples=10)
    response = client.post("/api/v1/training/jobs", json={})
    error = response.json()["error"]
    assert response.status_code == 422 and error["code"] == "TRAINING_DATA_INSUFFICIENT"
    assert "независимые сессии" in error["message"]


def test_background_job_versions_activation_rollback_and_predict(client: TestClient) -> None:
    build_dataset(client, sessions=3, samples=3)
    queued = client.post("/api/v1/training/jobs", json={"sample_kind": "static"})
    assert queued.status_code == 202 and queued.json()["status"] in {"queued", "running", "completed"}
    job = wait_job(client, queued.json()["id"])
    assert job["status"] == "completed" and job["model_id"] == "user-static-v1"
    metrics = job["metrics"]
    for key in (
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "confusion_matrix",
        "class_count",
        "session_count",
        "preprocessing",
        "dataset_version",
        "training_seconds",
    ):
        assert key in metrics
    assert metrics["test_sessions"] >= 2 and metrics["train_sessions"] >= 2
    assert (
        client.post("/api/v1/training/predict", json={"landmarks": landmarks("line")}).json()["error"]["code"]
        == "MODEL_NOT_READY"
    )

    assert client.post("/api/v1/training/models/user-static-v1/activate").json()["is_active"]
    hit = client.post("/api/v1/training/predict", json={"landmarks": landmarks("line")}).json()
    assert hit["state"] == "recognized" and hit["label"] == "Мой жест" and hit["top3"]
    negative = client.post("/api/v1/training/predict", json={"landmarks": landmarks("fan")}).json()
    assert negative["label"] is None and negative["is_negative"] and negative["state"] == "unknown"

    second = wait_job(
        client, client.post("/api/v1/training/jobs", json={"activate_on_success": True}).json()["id"]
    )
    assert second["model_id"] == "user-static-v2"
    models = {m["id"]: m for m in client.get("/api/v1/training/models").json()}
    assert models["user-static-v2"]["is_active"] and not models["user-static-v1"]["is_active"]
    rolled = client.post("/api/v1/training/models/rollback", params={"kind": "static"}).json()
    assert rolled["id"] == "user-static-v1" and rolled["is_active"]
    assert client.get("/api/v1/models", params={"type": "static_landmarks"}).json()[0]["source"] == "user"


def test_sequence_samples_and_training(client: TestClient) -> None:
    for name, shape, negative in (("Движение", "line", False), ("фон", "fan", True)):
        class_id = client.post(
            "/api/v1/training/classes", json={"name": name, "is_negative": negative}
        ).json()["id"]
        for _ in range(2):
            session_id = client.post("/api/v1/training/sessions", json={"class_id": class_id}).json()["id"]
            for _ in range(2):
                sequence = [landmarks(shape) for _ in range(6)]
                body = {"session_id": session_id, "kind": "sequence", "sequence": sequence}
                assert client.post("/api/v1/training/samples", json=body).json()["frame_count"] == 6
    job = wait_job(
        client, client.post("/api/v1/training/jobs", json={"sample_kind": "sequence"}).json()["id"]
    )
    assert job["status"] == "completed" and job["model_id"] == "user-sequence-v1"


def test_images_require_explicit_consent(client: TestClient, settings) -> None:  # type: ignore[no-untyped-def]
    class_id = client.post("/api/v1/training/classes", json={"name": "img"}).json()["id"]
    image = "data:image/jpeg;base64," + base64.b64encode(jpeg(RED)).decode()
    no_consent = client.post("/api/v1/training/sessions", json={"class_id": class_id}).json()["id"]
    refused = client.post(
        "/api/v1/training/samples",
        json={"session_id": no_consent, "landmarks": landmarks("line"), "image": image},
    )
    assert refused.status_code == 422 and "согласия" in refused.json()["error"]["message"]
    consent = client.post(
        "/api/v1/training/sessions", json={"class_id": class_id, "image_consent": True}
    ).json()["id"]
    sample = client.post(
        "/api/v1/training/samples",
        json={"session_id": consent, "landmarks": landmarks("line"), "image": image},
    ).json()
    assert sample["has_image"]
    stored = list(settings.training_image_dir.rglob("*.jpg"))
    assert len(stored) == 1
    client.delete(f"/api/v1/training/samples/{sample['id']}")
    assert not stored[0].exists()


def test_tampered_model_file_is_never_loaded(client: TestClient, settings) -> None:  # type: ignore[no-untyped-def]
    build_dataset(client, sessions=2, samples=3)
    job = wait_job(client, client.post("/api/v1/training/jobs", json={}).json()["id"])
    model_file = settings.user_model_root / job["model_id"] / "model.joblib"
    model_file.write_bytes(model_file.read_bytes() + b"tampered")
    response = client.post(f"/api/v1/training/models/{job['model_id']}/activate")
    assert response.status_code == 422 and response.json()["error"]["code"] == "MODEL_INVALID"
