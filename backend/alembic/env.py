"""Alembic environment.

Started by the backend at startup (a connection is passed in `config.attributes`)
or from the command line: `alembic upgrade head` inside backend/.
"""

from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import Connection

import app.db_models  # noqa: F401  - register tables on Base.metadata
from alembic import context
from app.core.config import get_settings
from app.database.base import Base
from app.database.engine import create_db_engine

config = context.config
target_metadata = Base.metadata


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection, target_metadata=target_metadata, render_as_batch=True, compare_type=True
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    if config.config_file_name is not None:
        fileConfig(config.config_file_name)
    engine = create_db_engine(get_settings().resolved_database_url)
    with engine.begin() as conn:
        _run(conn)
    engine.dispose()


def run_migrations_offline() -> None:
    context.configure(
        url=get_settings().resolved_database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
