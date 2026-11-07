from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

PACK = {
    "id": "pharmacy",
    "name": "Аптека",
    "version": "1.0",
    "enabled_labels": ["помогать", "врач"],
    "phrase_templates": [],
    "quick_responses": ["Рецепт есть?"],
}


def test_session_lifecycle(client: TestClient) -> None:
    created = client.post("/api/v1/sessions", json={"domain_pack_id": "clinic"})
    assert created.status_code == 201
    session = created.json()
    assert session["status"] == "active" and session["model_id"] == "tiny-sign"
    assert any(s["id"] == session["id"] for s in client.get("/api/v1/sessions").json())
    finished = client.post(f"/api/v1/sessions/{session['id']}/finish").json()
    assert (
        finished["status"] == "finished" and finished["end_reason"] == "api" and finished["duration_ms"] >= 0
    )
    assert client.get(f"/api/v1/sessions/{session['id']}").json()["events"] == []
    assert client.get("/api/v1/sessions/nope").json()["error"]["code"] == "SESSION_NOT_FOUND"
    assert (
        client.post("/api/v1/sessions", json={"domain_pack_id": "missing"}).json()["error"]["code"]
        == "DOMAIN_PACK_NOT_FOUND"
    )


def test_domain_pack_crud_and_compatibility(client: TestClient) -> None:
    listed = client.get("/api/v1/domain-packs").json()
    clinic = next(p for p in listed if p["id"] == "clinic")
    assert clinic["compatibility"] == {
        "model_id": "tiny-sign",
        "checked": True,
        "compatible": True,
        "missing_labels": [],
    }
    assert client.post("/api/v1/domain-packs", json=PACK).status_code == 201
    assert client.post("/api/v1/domain-packs", json=PACK).status_code == 409
    updated = client.put("/api/v1/domain-packs/pharmacy", json={**PACK, "version": "1.1"}).json()
    assert updated["version"] == "1.1"
    assert client.delete("/api/v1/domain-packs/pharmacy").status_code == 200
    assert client.get("/api/v1/domain-packs/pharmacy").json()["error"]["code"] == "DOMAIN_PACK_NOT_FOUND"


def test_pack_labels_must_exist_in_active_model(client: TestClient) -> None:
    response = client.post(
        "/api/v1/domain-packs", json={**PACK, "id": "bad", "enabled_labels": ["БОЛЬ", "врач"]}
    )
    error = response.json()["error"]
    assert error["code"] == "DOMAIN_PACK_INVALID" and error["details"]["missing_labels"] == ["БОЛЬ"]


def test_empty_or_inconsistent_packs_are_rejected(client: TestClient) -> None:
    assert client.post("/api/v1/domain-packs", json={**PACK, "enabled_labels": []}).status_code == 422
    bad_template = {**PACK, "phrase_templates": [{"tokens": ["голова"], "result": "x"}]}
    assert client.post("/api/v1/domain-packs", json=bad_template).status_code == 422


def test_corrupted_pack_is_listed_invalid_and_cannot_be_activated(
    client: TestClient, workspace: Path
) -> None:
    (workspace / "packs" / "broken.json").write_text("{not json", encoding="utf-8")
    listed = {p["id"]: p for p in client.get("/api/v1/domain-packs").json()}
    assert listed["broken"]["status"] == "invalid"
    response = client.put("/api/v1/settings", json={"active_domain_pack_id": "broken"})
    assert response.json()["error"]["code"] == "DOMAIN_PACK_INVALID"


def test_active_pack_cannot_be_deleted(client: TestClient) -> None:
    assert (
        client.put("/api/v1/settings", json={"active_domain_pack_id": "clinic"}).json()[
            "active_domain_pack_id"
        ]
        == "clinic"
    )
    assert client.delete("/api/v1/domain-packs/clinic").status_code == 409
    assert (
        client.put("/api/v1/settings", json={"active_domain_pack_id": ""}).json()["active_domain_pack_id"]
        is None
    )
