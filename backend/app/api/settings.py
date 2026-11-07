from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..dependencies import Container, get_container, get_db
from ..schemas.settings import SettingsOut, SettingsUpdate
from ..services import settings_service

router = APIRouter(prefix="/settings", tags=["settings"])


@router.get("", response_model=SettingsOut)
def get_settings(db: Session = Depends(get_db)) -> SettingsOut:
    return SettingsOut.model_validate(settings_service.get(db))


@router.put("", response_model=SettingsOut)
def update_settings(
    payload: SettingsUpdate, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> SettingsOut:
    row = settings_service.update(db, payload, container.packs, container.models.active())
    return SettingsOut.model_validate(row)
