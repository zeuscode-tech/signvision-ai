from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..dependencies import get_db
from ..schemas.lessons import AttemptCreate, AttemptOut, LessonOut
from ..services import lesson_service

router = APIRouter(tags=["lessons"])


@router.get("/lessons", response_model=list[LessonOut])
def list_lessons(db: Session = Depends(get_db)) -> list[LessonOut]:
    return lesson_service.list_lessons(db)


@router.get("/lessons/{lesson_id}", response_model=LessonOut)
def get_lesson(lesson_id: str, db: Session = Depends(get_db)) -> LessonOut:
    return lesson_service.get_lesson(db, lesson_id)


@router.post("/attempts", response_model=AttemptOut, status_code=201)
def create_attempt(payload: AttemptCreate, db: Session = Depends(get_db)) -> AttemptOut:
    return AttemptOut.model_validate(lesson_service.record_attempt(db, payload))
