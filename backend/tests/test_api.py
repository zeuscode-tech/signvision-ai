from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.database import Base, get_db
from app.main import app

engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
TestingSession = sessionmaker(bind=engine)
Base.metadata.create_all(engine)


def override_db():
    db = TestingSession()
    try: yield db
    finally: db.close()


app.dependency_overrides[get_db] = override_db
client = TestClient(app)


def valid_landmarks(offset=0.0):
    values=[]
    for i in range(21): values.extend([offset+i*.01, .2+i*.005, -i*.002])
    return values+[0.0]*63


def test_empty_state_and_class_session_sample_flow():
    assert client.get("/api/health").status_code == 200
    created = client.post("/api/classes", json={"name":"Тест", "description":"локальный", "is_unknown":False})
    assert created.status_code == 201
    class_id = created.json()["id"]
    session = client.post("/api/sessions", json={"gesture_class_id":class_id}).json()
    sample = client.post("/api/samples", json={"session_id":session["id"], "landmarks":valid_landmarks()})
    assert sample.status_code == 201
    rows = client.get(f'/api/sessions/{session["id"]}/samples').json()
    assert len(rows) == 1


def test_missing_hand_sample_is_rejected():
    class_id = client.post("/api/classes", json={"name":"Пустой", "description":"", "is_unknown":True}).json()["id"]
    session_id = client.post("/api/sessions", json={"gesture_class_id":class_id}).json()["id"]
    response = client.post("/api/samples", json={"session_id":session_id, "landmarks":[0]*126})
    assert response.status_code == 422


def test_attempt_updates_progress_and_review():
    result = client.post("/api/attempts", json={"gesture_name":"Тест", "mode":"practice", "success":False, "duration_seconds":4})
    assert result.status_code == 201
    progress = client.get("/api/progress").json()
    assert progress["attempts"] >= 1
    exported = client.get("/api/export").json()
    review = next(row for row in exported["reviews"] if row["gesture_name"] == "Тест")
    assert review["interval_days"] == 1


def test_lessons_are_honest_about_material_status():
    rows = client.get("/api/lessons").json()
    alphabet = next(x for x in rows if x["id"] == "alphabet")
    assert alphabet["material_status"] == "requires_verified_rsl_material"
    assert alphabet["exercises"] == []

