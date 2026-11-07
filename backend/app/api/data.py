from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from sqlalchemy.orm import Session

from ..core.timeutils import utcnow
from ..dependencies import Container, get_container, get_db
from ..schemas.data import DeleteResult, ImportResult

router = APIRouter(prefix="/data", tags=["user data"])


@router.get(
    "/export",
    response_class=Response,
    responses={200: {"content": {"application/zip": {}}, "description": "ZIP archive"}},
)
def export_data(
    include_models: bool = False, db: Session = Depends(get_db), container: Container = Depends(get_container)
) -> Response:
    archive = container.data.export(db, include_models)
    filename = f"signvision-export-{utcnow():%Y%m%d-%H%M%S}.zip"
    return Response(
        archive,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/import", response_model=ImportResult)
async def import_data(
    file: UploadFile = File(...), container: Container = Depends(get_container)
) -> ImportResult:
    path = await run_in_threadpool(container.data.save_upload, file.file)

    def run() -> ImportResult:
        try:
            with container.database.session() as db:
                return container.data.import_archive(db, path)
        finally:
            path.unlink(missing_ok=True)

    return await run_in_threadpool(run)


@router.delete("", response_model=DeleteResult)
def delete_data(
    confirm: str | None = Query(None, description="Must equal DELETE_ALL_USER_DATA"),
    db: Session = Depends(get_db),
    container: Container = Depends(get_container),
) -> DeleteResult:
    return container.data.delete_all(db, confirm)
