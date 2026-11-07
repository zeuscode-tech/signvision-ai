from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ..dependencies import Container, get_container, get_db
from ..schemas.sessions import SessionCreate, SessionDetail, SessionOut
from ..services import session_service

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionOut, status_code=201)
def create_session(
    payload: SessionCreate, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> SessionOut:
    if payload.domain_pack_id:
        container.packs.load(payload.domain_pack_id)  # 404 / 422 for missing or broken packs
    session = session_service.create(
        db,
        mode=payload.mode,
        model_id=payload.model_id or container.models.active_id,
        domain_pack_id=payload.domain_pack_id,
    )
    return SessionOut.model_validate(session)


@router.get("", response_model=list[SessionOut])
def list_sessions(
    limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), db: Session = Depends(get_db)
) -> list[SessionOut]:
    return [SessionOut.model_validate(s) for s in session_service.list_sessions(db, limit, offset)]


@router.get("/{session_id}", response_model=SessionDetail)
def get_session(session_id: str, db: Session = Depends(get_db)) -> SessionDetail:
    return SessionDetail.model_validate(session_service.get(db, session_id, with_events=True))


@router.post("/{session_id}/finish", response_model=SessionOut)
def finish_session(session_id: str, db: Session = Depends(get_db)) -> SessionOut:
    return SessionOut.model_validate(session_service.finish(db, session_id, reason="api"))
