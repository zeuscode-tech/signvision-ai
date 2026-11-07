"""Deprecated pre-v1 endpoints used by the current frontend (/api/*).

Thin adapters over the v1 services so the existing UI keeps working while the
SignBridge Live frontend moves to /api/v1 and /ws/recognition. No business
logic or SQL lives here.
"""

from __future__ import annotations

import base64
import binascii
from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..core.exceptions import AppError, ErrorCode
from ..core.timeutils import utcnow
from ..dependencies import Container, get_container, get_db
from ..recognition.hand_detection import NullHandDetector
from ..recognition.inference import infer_window, top_k
from ..recognition.preprocessing import FramePreprocessor, FrameTooLargeError, InvalidFrameError
from ..schemas.lessons import AttemptCreate
from ..schemas.training import (
    PredictIn,
    SampleCreate,
    TrainingClassCreate,
    TrainingJobCreate,
    TrainingSessionCreate,
)
from ..services import lesson_service
from ..services.model_service import RuntimeStatus

router = APIRouter(prefix="/api", tags=["legacy (deprecated)"], deprecated=True)

BUKVA_ID = "bukva-mobilenetv2-tsm"


class GestureCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=2000)
    is_unknown: bool = False


class LegacySessionCreate(BaseModel):
    gesture_class_id: int
    note: str = Field(default="", max_length=200)


class LandmarkPayload(BaseModel):
    landmarks: list[float] = Field(max_length=126)


class LegacySampleCreate(LandmarkPayload):
    session_id: int


class FrameSequencePayload(BaseModel):
    frames: list[str] = Field(min_length=1, max_length=64)


class LegacyAttempt(BaseModel):
    gesture_name: str = Field(min_length=1, max_length=120)
    mode: str = "practice"
    success: bool
    confidence: float | None = Field(default=None, ge=0, le=1)
    duration_seconds: int = Field(default=0, ge=0, le=86_400)


class LegacyImport(BaseModel):
    attempts: list[dict[str, Any]] = Field(default_factory=list, max_length=500_000)
    reviews: list[dict[str, Any]] = Field(default_factory=list, max_length=100_000)


def _class_dict(row: Any) -> dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "is_unknown": row.is_negative,
        "session_count": row.session_count,
        "sample_count": row.sample_count,
    }


@router.get("/health")
def health(db: Session = Depends(get_db), container: Container = Depends(get_container)) -> dict[str, Any]:
    active = container.training.active_model(db, "static")
    return {
        "status": "ok",
        "model_ready": active is not None,
        "trained_at": active[0].created_at if active else None,
        "bukva_ready": container.models.runtime_status(BUKVA_ID)
        in {RuntimeStatus.READY, RuntimeStatus.NOT_LOADED}
        and (container.settings.model_root / "bukva" / "bukva_mobilenetv2_tsm.onnx").is_file(),
    }


@router.get("/bukva/info")
async def bukva_info(container: Container = Depends(get_container)) -> dict[str, Any]:
    try:
        model = await run_in_threadpool(container.models.get, BUKVA_ID)
    except AppError:
        return {"ready": False, "source": "Bukva"}
    source = model.manifest.source or {}
    return {
        "ready": True,
        "source": "Bukva",
        "window_size": model.manifest.frame_count,
        "input_shape": model.manifest.input_shape,
        "classes": model.labels,
        "license": source.get("license"),
        "reported_top1": source.get("reported_top1"),
    }


@router.post("/bukva/predict")
async def bukva_predict(
    payload: FrameSequencePayload, container: Container = Depends(get_container)
) -> dict[str, Any]:
    model = await run_in_threadpool(container.models.get, BUKVA_ID)
    manifest = model.manifest
    if len(payload.frames) != manifest.frame_count:
        raise AppError(
            ErrorCode.VALIDATION_ERROR, f"Модель Bukva ожидает ровно {manifest.frame_count} кадров"
        )
    preprocessor = FramePreprocessor(
        manifest.preprocessing, NullHandDetector(), container.settings.max_frame_size_bytes
    )

    def run() -> dict[str, Any]:
        images = []
        for index, value in enumerate(payload.frames):
            encoded = value.split(",", 1)[1] if value.startswith("data:") else value
            try:
                images.append(preprocessor.process(base64.b64decode(encoded, validate=True), index).image)
            except (binascii.Error, InvalidFrameError) as exc:
                raise AppError(ErrorCode.INVALID_FRAME, "Некорректный JPEG-кадр") from exc
            except FrameTooLargeError as exc:
                raise AppError(ErrorCode.FRAME_TOO_LARGE) from exc
        output = infer_window(model, images, manifest.preprocessing)
        best = top_k(output.probabilities, model.labels, 3)
        threshold = manifest.confidence_threshold or container.settings.default_confidence_threshold
        accepted = best[0].confidence >= threshold
        return {
            "label": best[0].label if accepted else None,
            "candidate": best[0].label,
            "confidence": best[0].confidence,
            "accepted": accepted,
            "is_no_event": accepted and best[0].index == model.no_event_index,
            "top3": [{"label": c.label, "confidence": c.confidence} for c in best],
            "model": model.manifest.name,
            "note": f"Порог {threshold:.0%} — консервативное локальное правило отказа, "
            "а не опубликованная калибровка.",
        }

    return await run_in_threadpool(run)


@router.get("/lessons")
def lessons(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    return [lesson.model_dump() for lesson in lesson_service.list_lessons(db)]


@router.get("/classes")
def list_classes(
    db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> list[dict[str, Any]]:
    return [_class_dict(row) for row in container.training.list_classes(db)]


@router.post("/classes", status_code=201)
def create_class(
    payload: GestureCreate, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> dict[str, Any]:
    row = container.training.create_class(
        db,
        TrainingClassCreate(
            name=payload.name, description=payload.description, is_negative=payload.is_unknown
        ),
    )
    return _class_dict(row)


@router.delete("/classes/{class_id}")
def delete_class(
    class_id: int, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> dict[str, bool]:
    container.training.delete_class(db, class_id)
    return {"deleted": True}


@router.post("/sessions", status_code=201)
def create_session(
    payload: LegacySessionCreate, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> dict[str, Any]:
    row = container.training.create_session(
        db, TrainingSessionCreate(class_id=payload.gesture_class_id, note=payload.note)
    )
    return {"id": row.id, "gesture_class_id": row.class_id, "created_at": row.created_at}


@router.post("/samples", status_code=201)
def create_sample(
    payload: LegacySampleCreate, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> dict[str, Any]:
    row = container.training.add_sample(
        db, SampleCreate(session_id=payload.session_id, kind="static", landmarks=payload.landmarks)
    )
    return {"id": row.id, "session_id": row.session_id}


@router.get("/sessions/{session_id}/samples")
def list_samples(
    session_id: int, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> list[dict[str, Any]]:
    rows = container.training.list_samples(
        db, session_id=session_id, class_id=None, kind=None, limit=1000, offset=0
    )
    return [{"id": row.id, "created_at": row.created_at} for row in rows]


@router.delete("/samples/{sample_id}")
def delete_sample(
    sample_id: int, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> dict[str, bool]:
    container.training.delete_sample(db, sample_id)
    return {"deleted": True}


@router.post("/train")
async def train(container: Container = Depends(get_container)) -> dict[str, Any]:
    """Synchronous training + activation, as the old UI expects."""

    def run() -> dict[str, Any]:
        with container.database.session() as db:
            job = container.training.create_job(
                db, TrainingJobCreate(sample_kind="static", activate_on_success=True)
            )
        container.training.wait_for_job(job.id)
        with container.database.session() as db:
            result = container.training.get_job(db, job.id)
        if result.status != "completed" or result.metrics is None:
            raise AppError(ErrorCode(result.error_code or ErrorCode.TRAINING_FAILED), result.error_message)
        return _legacy_report(result.metrics, result.finished_at)

    return await run_in_threadpool(run)


def _legacy_report(metrics: dict[str, Any], trained_at: datetime | None) -> dict[str, Any]:
    per_class = metrics.get("per_class", {})
    return {
        "labels": metrics["labels"],
        "trained_at": trained_at,
        "sample_counts": metrics.get("samples_per_class"),
        "threshold": None,
        "metrics": {
            "labels": metrics["labels"],
            "precision": per_class.get("precision"),
            "recall": per_class.get("recall"),
            "f1": per_class.get("f1"),
            "support": per_class.get("support"),
            "confusion_matrix": metrics["confusion_matrix"],
            "test_samples": metrics.get("test_samples"),
            "split": metrics.get("split"),
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
        },
    }


@router.get("/model")
def model_info(
    db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> dict[str, Any]:
    active = container.training.active_model(db, "static")
    if active is None:
        return {"ready": False}
    entry, artifact = active
    out = container.training.model_out(entry)
    return {
        "ready": True,
        "model_id": entry.id,
        "threshold": artifact.get("threshold"),
        **_legacy_report(out.metrics or {"labels": [], "confusion_matrix": []}, entry.created_at),
    }


@router.post("/predict")
def predict(
    payload: LandmarkPayload, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> dict[str, Any]:
    result = container.training.predict(db, PredictIn(kind="static", landmarks=payload.landmarks))
    return {
        "label": result.label,
        "candidate": result.candidate,
        "confidence": result.confidence,
        "accepted": result.accepted,
        "message": "Распознано" if result.accepted else "Не удалось уверенно распознать",
    }


@router.post("/attempts", status_code=201)
def create_attempt(payload: LegacyAttempt, db: Session = Depends(get_db)) -> dict[str, Any]:
    mode = payload.mode if payload.mode in {"practice", "quiz", "recognition", "lesson"} else "practice"
    attempt = lesson_service.record_attempt(
        db,
        AttemptCreate(
            gesture_id=payload.gesture_name,
            mode=mode,
            success=payload.success,
            confidence=payload.confidence,
            duration_seconds=payload.duration_seconds,
        ),
    )
    return {"id": attempt.id, "created_at": attempt.created_at}


@router.get("/progress")
def progress(db: Session = Depends(get_db)) -> dict[str, Any]:
    data = lesson_service.progress(db)
    due = lesson_service.next_repetitions(db, 100)
    return {
        "attempts": data.attempts,
        "successes": data.successes,
        "practice_seconds": data.practice_seconds,
        "daily": [d.model_dump(mode="json") for d in data.daily],
        "reviews_due": [
            {
                "gesture_name": x.gesture_id,
                "due_date": x.due_date,
                "interval_days": x.interval_days,
                "repetitions": x.repetitions,
            }
            for x in due
        ],
    }


@router.get("/export")
def export_json(db: Session = Depends(get_db)) -> dict[str, Any]:
    return lesson_service.legacy_export(db) | {"schema_version": 1, "exported_at": utcnow()}


@router.post("/import")
def import_json(payload: LegacyImport, db: Session = Depends(get_db)) -> dict[str, int]:
    attempts = []
    for raw in payload.attempts:
        try:
            created = datetime.fromisoformat(raw["created_at"]) if "created_at" in raw else None
            attempts.append(
                (
                    AttemptCreate(
                        gesture_id=str(raw["gesture_name"]),
                        success=bool(raw["success"]),
                        confidence=raw.get("confidence"),
                        duration_seconds=int(raw.get("duration_seconds", 0)),
                    ),
                    created,
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise AppError(ErrorCode.IMPORT_FORMAT_INVALID, "Некорректная запись попытки") from exc
    reviews = []
    for raw in payload.reviews:
        try:
            reviews.append(
                {
                    "gesture_id": str(raw["gesture_name"]),
                    "repetitions": int(raw.get("repetitions", 0)),
                    "interval_days": int(raw.get("interval_days", 0)),
                    "easiness": max(1.3, float(raw.get("easiness", 2.5))),
                    "due_date": date.fromisoformat(raw["due_date"]) if raw.get("due_date") else None,
                    "last_quality": raw.get("last_quality"),
                }
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise AppError(ErrorCode.IMPORT_FORMAT_INVALID, "Некорректная запись повторения") from exc
    lesson_service.legacy_import(db, attempts, reviews)
    return {"imported_attempts": len(attempts), "imported_reviews": len(reviews)}


@router.delete("/data")
def delete_data(
    confirm: str | None = Query(None),
    db: Session = Depends(get_db),
    container: Container = Depends(get_container),
) -> dict[str, bool]:
    container.data.delete_all(db, confirm)
    return {"deleted": True}
