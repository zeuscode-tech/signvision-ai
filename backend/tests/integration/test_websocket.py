from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from app.recognition.hand_detection import HandObservation
from tests.conftest import BLUE, GRAY, RED, jpeg, make_settings, start_client

Message = dict[str, Any]


def read_until(ws: WebSocketTestSession, predicate: Callable[[Message], bool]) -> list[Message]:
    messages = []
    while True:
        message = ws.receive_json()
        messages.append(message)
        if predicate(message):
            return messages


def send_frames(ws: WebSocketTestSession, frame: bytes, count: int, delay: float = 0.03) -> None:
    for _ in range(count):
        ws.send_bytes(frame)
        time.sleep(delay)


def stop(ws: WebSocketTestSession) -> list[Message]:
    time.sleep(0.2)  # let the last inference finish
    ws.send_json({"type": "stop"})
    return read_until(ws, lambda m: m["type"] == "stopped")


def predictions(messages: list[Message]) -> list[Message]:
    return [m for m in messages if m["type"] == "prediction" and "inference_ms" in m]


def test_start_frames_prediction_and_metrics(client: TestClient) -> None:
    with client.websocket_connect("/ws/recognition") as ws:
        ws.send_json(
            {
                "type": "start",
                "mode": "isolated_sign",
                "domain_pack": None,
                "client": {"frame_width": 64, "frame_height": 48},
            }
        )
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["required_frames"] == 4 and ready["model_id"] == "tiny-sign"
        send_frames(ws, jpeg(RED), 14)
        messages = stop(ws)
    states = {m.get("state") for m in messages}
    assert {"waiting", "collecting", "processing", "recognized"} <= states
    accepted = [m for m in predictions(messages) if m["accepted"]]
    assert len(accepted) == 1, "a held gesture must produce exactly one word"
    word = accepted[0]
    assert word["label"] == "врач" and word["confidence"] > 0.9 and word["stable_count"] >= 2
    assert word["top3"][0]["label"] == "врач" and len(word["top3"]) == 3
    assert word["timestamp"].endswith("Z") and isinstance(word["event_id"], int)
    phrase = [m for m in messages if m["type"] == "phrase"][-1]
    assert phrase["tokens"] == ["врач"] and phrase["is_translation"] is False
    summary = messages[-1]["summary"]
    assert summary["accepted"] == 1 and summary["predictions"] >= 2 and summary["p95_latency_ms"] is not None

    session = client.get(f"/api/v1/sessions/{ready['session_id']}").json()
    assert session["status"] == "finished" and session["end_reason"] == "stop"
    assert session["accepted_count"] == 1 and session["frames_processed"] == 14
    assert session["median_latency_ms"] is not None and len(session["events"]) >= 2


def test_domain_pack_restricts_classes_and_uses_its_labels(client: TestClient) -> None:
    with client.websocket_connect("/ws/recognition") as ws:
        ws.send_json({"type": "start", "domain_pack": "clinic"})
        assert ws.receive_json()["domain_pack"] == "clinic"
        send_frames(ws, jpeg(BLUE), 10)  # "помогать" is not in the clinic pack
        send_frames(ws, jpeg(RED), 10)
        messages = stop(ws)
    results = predictions(messages)
    assert all(m["label"] != "помогать" for m in results)
    blue_results = [m for m in results if m["state"] == "uncertain"]
    assert blue_results and all(m["label"] is None and m["top3"] == [] for m in blue_results)
    assert [m["label"] for m in results if m["accepted"]] == ["ВРАЧ"]


def test_no_event_class_is_waiting_not_a_word(client: TestClient) -> None:
    with client.websocket_connect("/ws/recognition") as ws:
        ws.send_json({"type": "start"})
        ws.receive_json()
        send_frames(ws, jpeg(GRAY), 8)
        messages = stop(ws)
    results = predictions(messages)
    assert results and all(m["state"] == "waiting" and not m["accepted"] for m in results)
    assert messages[-1]["summary"]["accepted"] == 0


def test_invalid_and_oversized_frames_are_rejected(client: TestClient) -> None:
    with client.websocket_connect("/ws/recognition") as ws:
        ws.send_json({"type": "start"})
        ws.receive_json()
        ws.send_bytes(b"definitely not a jpeg" * 10)
        invalid = read_until(ws, lambda m: m["type"] == "error")[-1]
        ws.send_bytes(b"\xff\xd8\xff" + b"0" * 1_100_000)
        too_large = read_until(ws, lambda m: m["type"] == "error")[-1]
        summary = stop(ws)[-1]["summary"]
    assert invalid["code"] == "INVALID_FRAME" and too_large["code"] == "FRAME_TOO_LARGE"
    assert summary["invalid_frames"] == 2


def test_frame_rate_is_limited(workspace: Path) -> None:
    client = start_client(make_settings(workspace, max_websocket_fps=2))
    try:
        with client.websocket_connect("/ws/recognition") as ws:
            ws.send_json({"type": "start"})
            ws.receive_json()
            send_frames(ws, jpeg(RED), 12, delay=0)
            errors = read_until(ws, lambda m: m.get("code") == "RATE_LIMIT_EXCEEDED")
            summary = stop(ws)[-1]["summary"]
        assert errors[-1]["type"] == "error"
        assert summary["frames_dropped"] >= 8 and summary["frames_received"] == 12
    finally:
        client.__exit__(None, None, None)


def test_disconnect_finishes_session_and_releases_connection(client: TestClient) -> None:
    container = client.app.state.container  # type: ignore[attr-defined]
    with client.websocket_connect("/ws/recognition") as ws:
        ws.send_json({"type": "start"})
        session_id = ws.receive_json()["session_id"]
        send_frames(ws, jpeg(RED), 6)
        assert container.connections.count == 1
    deadline = time.time() + 5
    while time.time() < deadline:
        session = client.get(f"/api/v1/sessions/{session_id}").json()
        if session["status"] == "finished" and container.connections.count == 0:
            break
        time.sleep(0.05)
    assert session["status"] == "finished" and session["end_reason"] == "disconnect"
    assert session["frames_received"] == 6 and container.connections.count == 0


def test_origin_must_be_local_frontend(client: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as exc,
        client.websocket_connect("/ws/recognition", headers={"origin": "https://evil.example"}),
    ):
        pass
    assert exc.value.code == 1008
    with client.websocket_connect("/ws/recognition", headers={"origin": "http://127.0.0.1:5173"}) as ws:
        ws.send_json({"type": "start"})
        assert ws.receive_json()["type"] == "ready"


def test_first_message_must_be_start(client: TestClient) -> None:
    with client.websocket_connect("/ws/recognition") as ws:
        ws.send_json({"type": "hello"})
        assert ws.receive_json()["code"] == "VALIDATION_ERROR"


def test_unavailable_model_is_reported_honestly(workspace: Path) -> None:
    shutil.rmtree(workspace / "models" / "tiny")
    client = start_client(make_settings(workspace))
    try:
        with client.websocket_connect("/ws/recognition") as ws:
            ws.send_json({"type": "start"})
            message = ws.receive_json()
            assert message["state"] == "model_unavailable" and message["accepted"] is False
            assert message["label"] is None
    finally:
        client.__exit__(None, None, None)


def test_user_correction_is_stored_and_updates_phrase(client: TestClient) -> None:
    with client.websocket_connect("/ws/recognition") as ws:
        ws.send_json({"type": "start"})
        session_id = ws.receive_json()["session_id"]
        send_frames(ws, jpeg(RED), 8)
        accepted = read_until(ws, lambda m: m.get("accepted") is True)[-1]
        read_until(ws, lambda m: m["type"] == "phrase")
        ws.send_json({"type": "correct", "event_id": accepted["event_id"], "label": "голова"})
        corrected = read_until(ws, lambda m: m["type"] == "corrected")[-1]
        phrase = read_until(ws, lambda m: m["type"] == "phrase")[-1]
        ws.send_json({"type": "phrase_undo"})
        undone = read_until(ws, lambda m: m["type"] == "phrase")[-1]
        stop(ws)
    assert corrected["label"] == "голова" and phrase["tokens"] == ["голова"] and undone["tokens"] == []
    session = client.get(f"/api/v1/sessions/{session_id}").json()
    event = next(e for e in session["events"] if e["id"] == accepted["event_id"])
    assert event["corrected"] and event["selected_label"] == "голова" and event["predicted_label"] == "врач"
    assert session["corrections_count"] == 1


class ColourHandDetector:
    """Test double: coloured frames contain 'hands', gray frames do not."""

    available = True

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> HandObservation:
        spread = float(np.ptp(rgb.reshape(-1, 3).mean(axis=0)))
        present = spread > 30
        return HandObservation(present=present, boxes=[(0.2, 0.2, 0.8, 0.8)] if present else [])

    def close(self) -> None:
        return None


def test_hands_disappearing_ends_gesture_and_allows_repeat(client: TestClient, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    container = client.app.state.container  # type: ignore[attr-defined]
    monkeypatch.setattr(container.hands, "create", ColourHandDetector)
    with client.websocket_connect("/ws/recognition") as ws:
        ws.send_json({"type": "start"})
        assert ws.receive_json()["hand_detection"] is True
        send_frames(ws, jpeg(RED), 10)
        send_frames(ws, jpeg(GRAY), 15)  # hands gone for longer than PAUSE_MS
        send_frames(ws, jpeg(RED), 10)
        messages = stop(ws)
    assert "no_hands" in {m.get("state") for m in messages}
    assert [m["label"] for m in predictions(messages) if m["accepted"]] == ["врач", "врач"]


def test_short_gesture_after_pause_is_recognised(client: TestClient, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The window keeps sliding while hands are down, so a sign shorter than the window still completes it."""
    container = client.app.state.container  # type: ignore[attr-defined]
    monkeypatch.setattr(container.hands, "create", ColourHandDetector)
    client.put("/api/v1/settings", json={"stability_count": 1})
    with client.websocket_connect("/ws/recognition") as ws:
        ws.send_json({"type": "start"})
        ws.receive_json()
        send_frames(ws, jpeg(GRAY), 12)  # no hands: idle, but frames still fill the window
        send_frames(ws, jpeg(RED), 3)  # shorter than the 4-frame window
        messages = stop(ws)
    assert [m["label"] for m in predictions(messages) if m["accepted"]] == ["врач"]
