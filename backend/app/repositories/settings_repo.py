from __future__ import annotations

from sqlalchemy.orm import Session

from ..db_models import AppSettings

SETTINGS_ID = 1


def get_or_create(db: Session) -> AppSettings:
    row = db.get(AppSettings, SETTINGS_ID)
    if row is None:
        row = AppSettings(
            id=SETTINGS_ID, theme="system", language="ru", speech_enabled=False, speech_rate=1.0
        )
        db.add(row)
        db.flush()
    return row
