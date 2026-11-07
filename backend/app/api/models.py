from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from ..core.exceptions import AppError, ErrorCode
from ..db_models import ModelRegistryEntry
from ..dependencies import Container, get_container, get_db
from ..repositories import model_repo
from ..schemas.common import DeletedResponse
from ..schemas.models import ModelDetail, ModelOut, RegisterModelIn, ValidationReportOut
from ..services.model_service import ONNX_TYPE

router = APIRouter(prefix="/models", tags=["models"])


def to_out(entry: ModelRegistryEntry, container: Container) -> ModelOut:
    runtime = container.models.runtime_status(entry.id) if entry.type == ONNX_TYPE else entry.status
    return ModelOut(
        id=entry.id,
        name=entry.name,
        version=entry.version,
        type=entry.type,
        source=entry.source,
        manifest_path=entry.manifest_path,
        model_path=entry.model_path,
        labels_path=entry.labels_path,
        labels_count=entry.labels_count,
        preprocessing=json.loads(entry.preprocessing_json or "{}"),
        metrics=json.loads(entry.metrics_json) if entry.metrics_json else None,
        status=entry.status,
        status_message=entry.status_message,
        runtime_status=runtime,
        is_active=entry.is_active,
        activated_at=entry.activated_at,
        created_at=entry.created_at,
    )


def _entry(db: Session, model_id: str) -> ModelRegistryEntry:
    entry = model_repo.get(db, model_id)
    if entry is None:
        raise AppError(ErrorCode.MODEL_NOT_FOUND)
    return entry


@router.get("", response_model=list[ModelOut])
def list_models(
    type: str | None = None, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> list[ModelOut]:
    return [to_out(e, container) for e in model_repo.list_models(db, types=[type] if type else None)]


@router.get("/{model_id}", response_model=ModelDetail)
def get_model(
    model_id: str, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> ModelDetail:
    entry = _entry(db, model_id)
    return ModelDetail(
        **to_out(entry, container).model_dump(),
        labels=container.models.labels(model_id),
        manifest=container.models.manifest_dict(model_id) if entry.type == ONNX_TYPE else None,
    )


@router.post("/register", response_model=ModelOut, status_code=201)
async def register_model(payload: RegisterModelIn, container: Container = Depends(get_container)) -> ModelOut:
    entry = await run_in_threadpool(container.models.register, payload.manifest_path)
    return to_out(entry, container)


@router.post("/{model_id}/activate", response_model=ModelOut)
async def activate_model(model_id: str, container: Container = Depends(get_container)) -> ModelOut:
    with container.database.session() as db:
        entry_type = _entry(db, model_id).type
    if entry_type == ONNX_TYPE:
        await run_in_threadpool(container.models.activate, model_id)
    else:

        def activate_user() -> None:
            with container.database.session() as db:
                container.training.activate_model(db, model_id)

        await run_in_threadpool(activate_user)
    with container.database.session() as db:
        return to_out(_entry(db, model_id), container)


@router.post("/{model_id}/validate", response_model=ValidationReportOut)
async def validate_model(model_id: str, container: Container = Depends(get_container)) -> ValidationReportOut:
    report = await run_in_threadpool(container.models.validate, model_id)
    return ValidationReportOut.model_validate(report.to_dict())


@router.delete("/{model_id}", response_model=DeletedResponse)
def delete_model(model_id: str, container: Container = Depends(get_container)) -> DeletedResponse:
    container.models.delete(model_id)
    return DeletedResponse()
