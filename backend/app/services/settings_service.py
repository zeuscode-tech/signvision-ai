from __future__ import annotations

from sqlalchemy.orm import Session

from ..db_models import AppSettings
from ..recognition.model_loader import LoadedModel
from ..repositories import settings_repo
from ..schemas.settings import SettingsUpdate
from .domain_pack_service import DomainPackService


def get(db: Session) -> AppSettings:
    row = settings_repo.get_or_create(db)
    db.commit()
    return row


def update(
    db: Session, payload: SettingsUpdate, packs: DomainPackService, model: LoadedModel | None
) -> AppSettings:
    row = settings_repo.get_or_create(db)
    data = payload.model_dump(exclude_unset=True)
    if "active_domain_pack_id" in data:
        pack_id = data.pop("active_domain_pack_id") or None
        if pack_id is not None:
            pack = packs.load(pack_id)  # raises for missing / corrupted packs
            if model is not None:
                packs.resolve(pack.id, model)  # raises if labels are missing in the active model
        row.active_domain_pack_id = pack_id
    for key, value in data.items():
        setattr(row, key, value)
    db.commit()
    return row
