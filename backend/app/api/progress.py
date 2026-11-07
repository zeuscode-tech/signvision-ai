from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy.orm import Session

from ..dependencies import get_db
from ..schemas.lessons import ProgressOut, RepetitionResult, SrsItemOut
from ..services import lesson_service

router = APIRouter(tags=["progress"])


@router.get("/progress", response_model=ProgressOut)
def progress(db: Session = Depends(get_db)) -> ProgressOut:
    return lesson_service.progress(db)


@router.get("/repetition/next", response_model=list[SrsItemOut])
def next_repetition(limit: int = Query(10, ge=1, le=100), db: Session = Depends(get_db)) -> list[SrsItemOut]:
    return lesson_service.next_repetitions(db, limit)


@router.post("/repetition/{gesture_id}/result", response_model=SrsItemOut)
def repetition_result(
    payload: RepetitionResult,
    gesture_id: str = Path(min_length=1, max_length=120),
    db: Session = Depends(get_db),
) -> SrsItemOut:
    return lesson_service.record_repetition(db, gesture_id, payload.quality)
