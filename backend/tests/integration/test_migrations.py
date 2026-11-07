from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import inspect, text

import app.db_models  # noqa: F401
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from app.database.base import Base
from app.database.engine import create_db_engine
from app.database.migrations import downgrade_to_base, upgrade_to_head


def test_migrations_match_models_and_roundtrip(tmp_path: Path) -> None:
    engine = create_db_engine(f"sqlite:///{tmp_path / 'm.db'}")
    upgrade_to_head(engine)
    with engine.connect() as connection:
        assert (
            compare_metadata(
                MigrationContext.configure(connection, opts={"compare_type": True}), Base.metadata
            )
            == []
        )
        assert connection.execute(text("PRAGMA foreign_keys")).scalar() == 1
    downgrade_to_base(engine)
    assert set(inspect(engine).get_table_names()) <= {"alembic_version"}
    upgrade_to_head(engine)
    assert "training_jobs" in inspect(engine).get_table_names()
    engine.dispose()


def test_legacy_schema_data_is_migrated(tmp_path: Path) -> None:
    engine = create_db_engine(f"sqlite:///{tmp_path / 'old.db'}")
    features = json.dumps([0.1] * 126)
    with engine.begin() as c:
        c.execute(
            text(
                "CREATE TABLE gesture_classes (id INTEGER PRIMARY KEY, name VARCHAR(80), description TEXT, "
                "is_unknown BOOLEAN, created_at DATETIME)"
            )
        )
        c.execute(
            text(
                "CREATE TABLE capture_sessions (id INTEGER PRIMARY KEY, gesture_class_id INTEGER, "
                "created_at DATETIME, note VARCHAR(200))"
            )
        )
        c.execute(
            text(
                "CREATE TABLE training_samples (id INTEGER PRIMARY KEY, session_id INTEGER, "
                "features_json TEXT, created_at DATETIME)"
            )
        )
        c.execute(text("CREATE INDEX ix_training_samples_session_id ON training_samples (session_id)"))
        c.execute(
            text(
                "CREATE TABLE attempts (id INTEGER PRIMARY KEY, gesture_name VARCHAR(80), mode VARCHAR(20), "
                "success BOOLEAN, confidence FLOAT, duration_seconds INTEGER, created_at DATETIME)"
            )
        )
        c.execute(
            text(
                "CREATE TABLE review_states (id INTEGER PRIMARY KEY, gesture_name VARCHAR(80), "
                "repetitions INTEGER, interval_days INTEGER, easiness FLOAT, due_date DATE, "
                "last_quality INTEGER)"
            )
        )
        c.execute(text("INSERT INTO gesture_classes VALUES (1, 'Привет', '', 0, '2025-01-01 10:00:00')"))
        c.execute(text("INSERT INTO capture_sessions VALUES (1, 1, '2025-01-01 10:00:00', 'n')"))
        c.execute(
            text("INSERT INTO training_samples VALUES (1, 1, :f, '2025-01-01 10:00:00')"), {"f": features}
        )
        c.execute(
            text("INSERT INTO attempts VALUES (1, 'Привет', 'practice', 1, 0.9, 3, '2025-01-01 10:00:00')")
        )
        c.execute(text("INSERT INTO review_states VALUES (1, 'Привет', 1, 1, 2.6, '2025-01-02', 5)"))
    upgrade_to_head(engine)
    with engine.connect() as c:
        assert c.execute(text("SELECT name, normalized_name FROM training_classes")).one() == (
            "Привет",
            "привет",
        )
        assert c.execute(text("SELECT kind, features_json FROM training_samples")).one() == (
            "static",
            features,
        )
        assert c.execute(text("SELECT gesture_id FROM lesson_attempts")).scalar() == "Привет"
        assert c.execute(text("SELECT interval_days FROM srs_items")).scalar() == 1
    assert "gesture_classes" not in inspect(engine).get_table_names()
    engine.dispose()


def test_interrupted_legacy_migration_is_resumed_without_losing_samples(tmp_path: Path) -> None:
    engine = create_db_engine(f"sqlite:///{tmp_path / 'interrupted.db'}")
    features = json.dumps([0.2] * 126)
    with engine.begin() as c:
        c.execute(
            text(
                "CREATE TABLE gesture_classes (id INTEGER PRIMARY KEY, name VARCHAR(80), description TEXT, "
                "is_unknown BOOLEAN, created_at DATETIME)"
            )
        )
        c.execute(
            text(
                "CREATE TABLE capture_sessions (id INTEGER PRIMARY KEY, gesture_class_id INTEGER, "
                "created_at DATETIME, note VARCHAR(200))"
            )
        )
        c.execute(
            text(
                "CREATE TABLE legacy_training_samples (id INTEGER PRIMARY KEY, session_id INTEGER, "
                "features_json TEXT, created_at DATETIME)"
            )
        )
        c.execute(
            text("CREATE INDEX ix_training_samples_session_id ON legacy_training_samples (session_id)")
        )
        c.execute(text("CREATE TABLE training_samples (id INTEGER PRIMARY KEY, session_id INTEGER)"))
        c.execute(text("CREATE TABLE settings (id INTEGER PRIMARY KEY)"))
        c.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) PRIMARY KEY)"))
        c.execute(text("INSERT INTO gesture_classes VALUES (1, 'Помощь', '', 0, '2025-01-01')"))
        c.execute(text("INSERT INTO capture_sessions VALUES (1, 1, '2025-01-01', '')"))
        c.execute(
            text("INSERT INTO legacy_training_samples VALUES (1, 1, :f, '2025-01-01')"),
            {"f": features},
        )

    upgrade_to_head(engine)

    with engine.connect() as c:
        assert c.execute(text("SELECT name FROM training_classes")).scalar() == "Помощь"
        assert c.execute(text("SELECT features_json FROM training_samples")).scalar() == features
        assert c.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0001"
    assert "legacy_training_samples" not in inspect(engine).get_table_names()
    engine.dispose()
