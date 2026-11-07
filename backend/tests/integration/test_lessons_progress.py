from __future__ import annotations

from fastapi.testclient import TestClient


def test_lessons_seeded_from_catalogue(client: TestClient) -> None:
    lessons = client.get("/api/v1/lessons").json()
    assert [lesson["level"] for lesson in lessons] == sorted(lesson["level"] for lesson in lessons)
    assert client.get("/api/v1/lessons/basics").json()["exercises"][0]["verified"] is True
    assert client.get("/api/v1/lessons/nope").json()["error"]["code"] == "LESSON_NOT_FOUND"


def test_attempts_progress_and_spaced_repetition(client: TestClient) -> None:
    for success in (True, True, False):
        assert (
            client.post(
                "/api/v1/attempts",
                json={"lesson_id": "basics", "gesture_id": "врач", "success": success, "duration_seconds": 5},
            ).status_code
            == 201
        )
    progress = client.get("/api/v1/progress").json()
    assert progress["attempts"] == 3 and progress["successes"] == 2 and progress["practice_seconds"] == 15
    assert progress["gestures"][0]["gesture_id"] == "врач" and progress["srs_items"] == 1
    assert client.get("/api/v1/repetition/next").json() == []  # failed review → due tomorrow
    item = client.post("/api/v1/repetition/голова/result", json={"quality": 4}).json()
    assert item["repetitions"] == 1 and item["interval_days"] == 1
    assert client.post("/api/v1/repetition/голова/result", json={"quality": 9}).status_code == 422
    assert (
        client.post(
            "/api/v1/attempts", json={"lesson_id": "missing", "gesture_id": "x", "success": True}
        ).status_code
        == 404
    )
