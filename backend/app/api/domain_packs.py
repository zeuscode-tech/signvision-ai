from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..dependencies import Container, get_container, get_db
from ..schemas.common import DeletedResponse
from ..schemas.domain_packs import DomainPackBroken, DomainPackIn, DomainPackOut
from ..services import settings_service

router = APIRouter(prefix="/domain-packs", tags=["domain packs"])


def _active_pack(db: Session) -> str | None:
    return settings_service.get(db).active_domain_pack_id


@router.get("", response_model=list[DomainPackOut | DomainPackBroken])
def list_packs(
    db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> list[DomainPackOut | DomainPackBroken]:
    return container.packs.list(container.models.active(), _active_pack(db))


@router.get("/{pack_id}", response_model=DomainPackOut)
def get_pack(
    pack_id: str, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> DomainPackOut:
    return container.packs.to_out(container.packs.load(pack_id), container.models.active(), _active_pack(db))


@router.post("", response_model=DomainPackOut, status_code=201)
def create_pack(
    payload: DomainPackIn, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> DomainPackOut:
    model = container.models.active()
    return container.packs.to_out(container.packs.create(payload, model), model, _active_pack(db))


@router.put("/{pack_id}", response_model=DomainPackOut)
def update_pack(
    pack_id: str,
    payload: DomainPackIn,
    db: Session = Depends(get_db),
    container: Container = Depends(get_container),
) -> DomainPackOut:
    model = container.models.active()
    return container.packs.to_out(container.packs.update(pack_id, payload, model), model, _active_pack(db))


@router.delete("/{pack_id}", response_model=DeletedResponse)
def delete_pack(
    pack_id: str, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> DeletedResponse:
    container.packs.delete(pack_id, _active_pack(db))
    return DeletedResponse()
