"""Run Alembic migrations programmatically at startup."""

from __future__ import annotations

import logging

from sqlalchemy import Engine

from alembic import command
from alembic.config import Config

from ..core.config import BACKEND_ROOT

logger = logging.getLogger(__name__)


def alembic_config() -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    return config


def upgrade_to_head(engine: Engine) -> None:
    config = alembic_config()
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
    logger.info("Database schema is up to date")


def downgrade_to_base(engine: Engine) -> None:
    config = alembic_config()
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.downgrade(config, "base")
