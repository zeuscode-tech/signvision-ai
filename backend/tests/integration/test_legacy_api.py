"""The current frontend still calls the pre-v1 /api/* endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.integration.test_training_api import landmarks


def test_studio_flow_train_and_predict(client: TestClient) -> None:
    assert client.get("/api/health").json()["model_ready"] is False
    for name, shape, unknown in (("Тест", "line", False), ("неизвестно", "fan", True)):
        class_id = client.post(
            "/api/classes", json={"name": name, "description": "", "is_unknown": unknown}
        ).json()["id"]
        for _ in range(2):
            session = client.post("/api/sessions", json={"gesture_class_id": class_id}).json()
            for _ in range(3):
                assert (
                    client.post(
                        "/api/samples", json={"session_id": session["id"], "landmarks": landmarks(shape)}
                    ).status_code
                    == 201
                )
            assert len(client.get(f"/api/sessions/{session['id']}/samples").json()) == 3
    classes = client.get("/api/classes").json()
    assert {c["name"]: c["sample_count"] for c in classes} == {"Тест": 6, "неизвестно": 6}
    report = client.post("/api/train").json()
    assert report["metrics"]["confusion_matrix"] and report["metrics"]["test_samples"] > 0
    assert client.get("/api/model").json()["ready"] is True
    prediction = client.post("/api/predict", json={"landmarks": landmarks("line")}).json()
    assert prediction["accepted"] and prediction["label"] == "Тест"


def test_missing_hand_sample_rejected(client: TestClient) -> None:
    class_id = client.post("/api/classes", json={"name": "Пустой", "is_unknown": True}).json()["id"]
    session_id = client.post("/api/sessions", json={"gesture_class_id": class_id}).json()["id"]
    assert (
        client.post("/api/samples", json={"session_id": session_id, "landmarks": [0] * 126}).status_code
        == 422
    )


def test_progress_export_import_and_delete(client: TestClient) -> None:
    assert (
        client.post(
            "/api/attempts",
            json={"gesture_name": "Тест", "mode": "practice", "success": False, "duration_seconds": 4},
        ).status_code
        == 201
    )
    progress = client.get("/api/progress").json()
    assert progress["attempts"] == 1 and progress["reviews_due"] == []
    exported = client.get("/api/export").json()
    review = next(row for row in exported["reviews"] if row["gesture_name"] == "Тест")
    assert review["interval_days"] == 1 and exported["schema_version"] == 1
    assert (
        client.post(
            "/api/import", json={"attempts": exported["attempts"], "reviews": exported["reviews"]}
        ).json()["imported_attempts"]
        == 1
    )
    assert client.get("/api/progress").json()["attempts"] == 2
    assert client.delete("/api/data").status_code == 400
    assert client.delete("/api/data", params={"confirm": "DELETE_ALL_USER_DATA"}).json() == {"deleted": True}


def test_lessons_are_honest_about_material_status(client: TestClient) -> None:
    rows = client.get("/api/lessons").json()
    alphabet = next(x for x in rows if x["id"] == "alphabet")
    assert alphabet["material_status"] == "requires_verified_rsl_material" and alphabet["exercises"] == []
    assert client.get("/api/bukva/info").json() == {"ready": False, "source": "Bukva"}
