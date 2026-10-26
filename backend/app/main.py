from __future__ import annotations
from collections import defaultdict
from datetime import date, datetime, timedelta
import json
import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import BUKVA_MODEL_PATH, MODEL_PATH
from .database import Base, engine, get_db
from .lessons import LEVELS
from .models import Attempt, CaptureSession, GestureClass, ReviewState, TrainingSample
from .schemas import AttemptCreate, FrameSequencePayload, GestureCreate, ImportPayload, LandmarkPayload, SampleCreate, SessionCreate
from .ml.features import extract_features
from .ml.classifier import load_artifact, predict as model_predict, train_classifier
from .ml.bukva import model_info as bukva_model_info, predict as bukva_predict

app = FastAPI(title="SignVision AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(engine)


def gesture_dict(item: GestureClass, db: Session) -> dict:
    session_count = db.scalar(select(func.count(CaptureSession.id)).where(CaptureSession.gesture_class_id == item.id)) or 0
    sample_count = db.scalar(
        select(func.count(TrainingSample.id))
        .join(CaptureSession)
        .where(CaptureSession.gesture_class_id == item.id)
    ) or 0
    return {"id": item.id, "name": item.name, "description": item.description, "is_unknown": item.is_unknown,
            "session_count": session_count, "sample_count": sample_count}


@app.get("/api/health")
def health():
    artifact = load_artifact(MODEL_PATH)
    return {"status": "ok", "model_ready": artifact is not None,
            "trained_at": artifact.get("trained_at") if artifact else None,
            "bukva_ready": BUKVA_MODEL_PATH.exists()}


@app.get("/api/bukva/info")
def bukva_info():
    try:
        return bukva_model_info(BUKVA_MODEL_PATH)
    except Exception as exc:
        raise HTTPException(500, f"Не удалось загрузить Bukva: {exc}") from exc


@app.post("/api/bukva/predict")
def predict_bukva(payload: FrameSequencePayload):
    try:
        return bukva_predict(BUKVA_MODEL_PATH, payload.frames)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Ошибка инференса Bukva: {exc}") from exc


@app.get("/api/lessons")
def lessons():
    return LEVELS


@app.get("/api/classes")
def list_classes(db: Session = Depends(get_db)):
    return [gesture_dict(row, db) for row in db.scalars(select(GestureClass).order_by(GestureClass.created_at)).all()]


@app.post("/api/classes", status_code=201)
def create_class(payload: GestureCreate, db: Session = Depends(get_db)):
    if db.scalar(select(GestureClass).where(func.lower(GestureClass.name) == payload.name.lower())):
        raise HTTPException(409, "Класс с таким названием уже существует")
    item = GestureClass(**payload.model_dump())
    db.add(item); db.commit(); db.refresh(item)
    return gesture_dict(item, db)


@app.delete("/api/classes/{class_id}")
def delete_class(class_id: int, db: Session = Depends(get_db)):
    item = db.get(GestureClass, class_id)
    if not item: raise HTTPException(404, "Класс не найден")
    db.delete(item); db.commit()
    return {"deleted": True}


@app.post("/api/sessions", status_code=201)
def create_session(payload: SessionCreate, db: Session = Depends(get_db)):
    if not db.get(GestureClass, payload.gesture_class_id): raise HTTPException(404, "Класс не найден")
    item = CaptureSession(**payload.model_dump())
    db.add(item); db.commit(); db.refresh(item)
    return {"id": item.id, "gesture_class_id": item.gesture_class_id, "created_at": item.created_at}


@app.post("/api/samples", status_code=201)
def create_sample(payload: SampleCreate, db: Session = Depends(get_db)):
    if not db.get(CaptureSession, payload.session_id): raise HTTPException(404, "Сессия не найдена")
    try: features = extract_features(payload.landmarks)
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc
    item = TrainingSample(session_id=payload.session_id, features_json=json.dumps(features.tolist()))
    db.add(item); db.commit(); db.refresh(item)
    return {"id": item.id, "session_id": item.session_id}


@app.get("/api/sessions/{session_id}/samples")
def list_samples(session_id: int, db: Session = Depends(get_db)):
    rows = db.scalars(select(TrainingSample).where(TrainingSample.session_id == session_id).order_by(TrainingSample.id.desc())).all()
    return [{"id": row.id, "created_at": row.created_at} for row in rows]


@app.delete("/api/samples/{sample_id}")
def delete_sample(sample_id: int, db: Session = Depends(get_db)):
    item = db.get(TrainingSample, sample_id)
    if not item: raise HTTPException(404, "Образец не найден")
    db.delete(item); db.commit(); return {"deleted": True}


@app.post("/api/train")
def train(db: Session = Depends(get_db)):
    rows = db.execute(
        select(TrainingSample, CaptureSession, GestureClass)
        .join(CaptureSession, TrainingSample.session_id == CaptureSession.id)
        .join(GestureClass, CaptureSession.gesture_class_id == GestureClass.id)
    ).all()
    if not rows: raise HTTPException(422, "Нет обучающих образцов")
    X = np.asarray([json.loads(sample.features_json) for sample, _, _ in rows], dtype=np.float32)
    y = np.asarray([gesture.name for _, _, gesture in rows])
    groups = np.asarray([session.id for _, session, _ in rows])
    try: return train_classifier(X, y, groups, MODEL_PATH)
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc


@app.get("/api/model")
def model_info():
    artifact = load_artifact(MODEL_PATH)
    if not artifact: return {"ready": False}
    return {"ready": True, **{k: v for k, v in artifact.items() if k != "model"}}


@app.post("/api/predict")
def predict(payload: LandmarkPayload):
    artifact = load_artifact(MODEL_PATH)
    if not artifact: raise HTTPException(409, "Модель ещё не обучена")
    try: features = extract_features(payload.landmarks)
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc
    return model_predict(artifact, features)


def update_review(db: Session, name: str, success: bool) -> None:
    state = db.scalar(select(ReviewState).where(ReviewState.gesture_name == name))
    if not state:
        # SQLAlchemy column defaults are applied on INSERT, so initialize values
        # explicitly because SM-2 calculations happen before the first flush.
        state = ReviewState(
            gesture_name=name,
            repetitions=0,
            interval_days=0,
            easiness=2.5,
            due_date=date.today(),
        )
        db.add(state)
    quality = 5 if success else 2
    if quality < 3:
        state.repetitions = 0; state.interval_days = 1
    else:
        state.repetitions += 1
        state.interval_days = 1 if state.repetitions == 1 else 6 if state.repetitions == 2 else max(1, round(state.interval_days * state.easiness))
    state.easiness = max(1.3, state.easiness + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
    state.due_date = date.today() + timedelta(days=state.interval_days)
    state.last_quality = quality


@app.post("/api/attempts", status_code=201)
def create_attempt(payload: AttemptCreate, db: Session = Depends(get_db)):
    item = Attempt(**payload.model_dump()); db.add(item)
    update_review(db, payload.gesture_name, payload.success)
    db.commit(); db.refresh(item)
    return {"id": item.id, "created_at": item.created_at}


@app.get("/api/progress")
def progress(db: Session = Depends(get_db)):
    attempts = db.scalars(select(Attempt).order_by(Attempt.created_at)).all()
    daily = defaultdict(lambda: {"attempts": 0, "successes": 0, "seconds": 0})
    for row in attempts:
        key = row.created_at.date().isoformat(); daily[key]["attempts"] += 1
        daily[key]["successes"] += int(row.success); daily[key]["seconds"] += row.duration_seconds
    reviews = db.scalars(select(ReviewState).order_by(ReviewState.due_date)).all()
    return {
        "attempts": len(attempts), "successes": sum(int(x.success) for x in attempts),
        "practice_seconds": sum(x.duration_seconds for x in attempts),
        "daily": [{"date": key, **value} for key, value in sorted(daily.items())],
        "reviews_due": [{"gesture_name": x.gesture_name, "due_date": x.due_date, "interval_days": x.interval_days,
                         "repetitions": x.repetitions} for x in reviews if x.due_date <= date.today()],
    }


@app.get("/api/export")
def export_data(db: Session = Depends(get_db)):
    attempts = db.scalars(select(Attempt)).all(); reviews = db.scalars(select(ReviewState)).all()
    return {"schema_version": 1, "exported_at": datetime.utcnow(),
            "attempts": [{"gesture_name": x.gesture_name, "mode": x.mode, "success": x.success,
                          "confidence": x.confidence, "duration_seconds": x.duration_seconds,
                          "created_at": x.created_at.isoformat()} for x in attempts],
            "reviews": [{"gesture_name": x.gesture_name, "repetitions": x.repetitions,
                         "interval_days": x.interval_days, "easiness": x.easiness,
                         "due_date": x.due_date.isoformat(), "last_quality": x.last_quality} for x in reviews]}


@app.post("/api/import")
def import_data(payload: ImportPayload, db: Session = Depends(get_db)):
    for raw in payload.attempts:
        allowed = {k: raw[k] for k in ("gesture_name", "mode", "success", "confidence", "duration_seconds") if k in raw}
        if "created_at" in raw: allowed["created_at"] = datetime.fromisoformat(raw["created_at"])
        db.add(Attempt(**allowed))
    for raw in payload.reviews:
        name = raw.get("gesture_name")
        if not name: continue
        state = db.scalar(select(ReviewState).where(ReviewState.gesture_name == name)) or ReviewState(gesture_name=name)
        for key in ("repetitions", "interval_days", "easiness", "last_quality"):
            if key in raw: setattr(state, key, raw[key])
        if "due_date" in raw: state.due_date = date.fromisoformat(raw["due_date"])
        db.add(state)
    db.commit(); return {"imported_attempts": len(payload.attempts), "imported_reviews": len(payload.reviews)}


@app.delete("/api/data")
def delete_user_data(db: Session = Depends(get_db)):
    for table in (TrainingSample, CaptureSession, GestureClass, Attempt, ReviewState): db.query(table).delete()
    db.commit()
    if MODEL_PATH.exists(): MODEL_PATH.unlink()
    return {"deleted": True}

