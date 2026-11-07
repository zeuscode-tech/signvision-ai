from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..dependencies import Container, get_container, get_db
from ..schemas.common import DeletedResponse
from ..schemas.training import (
    PredictIn,
    PredictOut,
    SampleCreate,
    SampleOut,
    TrainingClassCreate,
    TrainingClassOut,
    TrainingJobCreate,
    TrainingJobOut,
    TrainingModelOut,
    TrainingSessionCreate,
    TrainingSessionOut,
)

router = APIRouter(prefix="/training", tags=["training studio"])


@router.post("/classes", response_model=TrainingClassOut, status_code=201)
def create_class(
    payload: TrainingClassCreate, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> TrainingClassOut:
    return container.training.create_class(db, payload)


@router.get("/classes", response_model=list[TrainingClassOut])
def list_classes(
    db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> list[TrainingClassOut]:
    return container.training.list_classes(db)


@router.delete("/classes/{class_id}", response_model=DeletedResponse)
def delete_class(
    class_id: int, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> DeletedResponse:
    container.training.delete_class(db, class_id)
    return DeletedResponse()


@router.post("/sessions", response_model=TrainingSessionOut, status_code=201)
def create_session(
    payload: TrainingSessionCreate,
    db: Session = Depends(get_db),
    container: Container = Depends(get_container),
) -> TrainingSessionOut:
    return container.training.create_session(db, payload)


@router.get("/sessions", response_model=list[TrainingSessionOut])
def list_sessions(
    class_id: int | None = None, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> list[TrainingSessionOut]:
    return container.training.list_sessions(db, class_id)


@router.post("/samples", response_model=SampleOut, status_code=201)
def create_sample(
    payload: SampleCreate, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> SampleOut:
    return container.training.add_sample(db, payload)


@router.get("/samples", response_model=list[SampleOut])
def list_samples(
    session_id: int | None = None,
    class_id: int | None = None,
    kind: Literal["static", "sequence"] | None = None,
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    container: Container = Depends(get_container),
) -> list[SampleOut]:
    return container.training.list_samples(
        db, session_id=session_id, class_id=class_id, kind=kind, limit=limit, offset=offset
    )


@router.delete("/samples/{sample_id}", response_model=DeletedResponse)
def delete_sample(
    sample_id: int, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> DeletedResponse:
    container.training.delete_sample(db, sample_id)
    return DeletedResponse()


@router.post("/jobs", response_model=TrainingJobOut, status_code=202)
def create_job(
    payload: TrainingJobCreate, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> TrainingJobOut:
    return container.training.create_job(db, payload)


@router.get("/jobs/{job_id}", response_model=TrainingJobOut)
def get_job(
    job_id: str, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> TrainingJobOut:
    return container.training.get_job(db, job_id)


@router.get("/models", response_model=list[TrainingModelOut])
def list_models(
    db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> list[TrainingModelOut]:
    return container.training.list_models(db)


@router.post("/models/rollback", response_model=TrainingModelOut)
def rollback(
    kind: Literal["static", "sequence"] = "static",
    db: Session = Depends(get_db),
    container: Container = Depends(get_container),
) -> TrainingModelOut:
    return container.training.rollback(db, kind)


@router.post("/models/{model_id}/activate", response_model=TrainingModelOut)
def activate_model(
    model_id: str, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> TrainingModelOut:
    return container.training.activate_model(db, model_id)


@router.post("/predict", response_model=PredictOut)
def predict(
    payload: PredictIn, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> PredictOut:
    return container.training.predict(db, payload)
