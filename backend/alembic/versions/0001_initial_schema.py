"""Initial SignBridge schema.

Also migrates data from the pre-Alembic SignVision schema (gesture_classes,
capture_sessions, training_samples, attempts, review_states) when it exists.

Revision ID: 0001
Revises:
Create Date: 2026-11-07
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_NEW_TABLES_REVERSE = (
    "training_jobs",
    "training_samples",
    "training_sessions",
    "training_classes",
    "srs_items",
    "lesson_attempts",
    "lessons",
    "recognition_events",
    "recognition_sessions",
    "model_registry",
    "settings",
)


def _recover_interrupted_legacy_upgrade(bind: sa.Connection, existing: set[str]) -> set[str]:
    """Remove only the empty new schema left by an interrupted legacy migration.

    SQLite DDL is non-transactional. If the first migration failed after renaming
    the old samples table, the legacy tables still contain the user's data while
    some new empty tables remain. Re-running the migration must clean those new
    tables before trying again instead of asking the user to delete the database.
    """
    if "gesture_classes" not in existing or "legacy_training_samples" not in existing:
        return existing
    for table in _NEW_TABLES_REVERSE:
        if table in existing:
            bind.execute(sa.text(f'DROP TABLE "{table}"'))  # names are fixed above
    return set(sa.inspect(bind).get_table_names())


def upgrade() -> None:
    bind = op.get_bind()
    existing = set(sa.inspect(bind).get_table_names())
    existing = _recover_interrupted_legacy_upgrade(bind, existing)
    legacy = "gesture_classes" in existing
    if legacy and "legacy_training_samples" not in existing:
        # The old schema used the same name for a table with different columns.
        op.rename_table("training_samples", "legacy_training_samples")
    if legacy:
        # SQLite keeps explicit index names when a table is renamed. The old index
        # would otherwise collide with the identically named index on the new table.
        op.execute("DROP INDEX IF EXISTS ix_training_samples_session_id")

    op.create_table(
        "settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("theme", sa.String(16), nullable=False),
        sa.Column("language", sa.String(16), nullable=False),
        sa.Column("active_model_id", sa.String(64), nullable=True),
        sa.Column("active_domain_pack_id", sa.String(64), nullable=True),
        sa.Column("confidence_threshold", sa.Float(), nullable=True),
        sa.Column("stability_count", sa.Integer(), nullable=True),
        sa.Column("cooldown_ms", sa.Integer(), nullable=True),
        sa.Column("speech_enabled", sa.Boolean(), nullable=False),
        sa.Column("speech_rate", sa.Float(), nullable=False),
        sa.Column("speech_voice", sa.String(120), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_settings"),
    )
    op.create_table(
        "model_registry",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("version", sa.String(40), nullable=False),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("manifest_path", sa.String(500), nullable=True),
        sa.Column("model_path", sa.String(500), nullable=False),
        sa.Column("labels_path", sa.String(500), nullable=True),
        sa.Column("labels_count", sa.Integer(), nullable=False),
        sa.Column("preprocessing_json", sa.Text(), nullable=False),
        sa.Column("metrics_json", sa.Text(), nullable=True),
        sa.Column("file_sha256", sa.String(64), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("status_message", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("activated_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_model_registry"),
    )
    op.create_index("ix_model_registry_type", "model_registry", ["type"])

    op.create_table(
        "recognition_sessions",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("mode", sa.String(32), nullable=False),
        sa.Column("model_id", sa.String(64), nullable=True),
        sa.Column("domain_pack_id", sa.String(64), nullable=True),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("end_reason", sa.String(32), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("frames_received", sa.Integer(), nullable=False),
        sa.Column("frames_processed", sa.Integer(), nullable=False),
        sa.Column("frames_dropped", sa.Integer(), nullable=False),
        sa.Column("invalid_frames", sa.Integer(), nullable=False),
        sa.Column("predictions_count", sa.Integer(), nullable=False),
        sa.Column("accepted_count", sa.Integer(), nullable=False),
        sa.Column("corrections_count", sa.Integer(), nullable=False),
        sa.Column("avg_latency_ms", sa.Float(), nullable=True),
        sa.Column("median_latency_ms", sa.Float(), nullable=True),
        sa.Column("p95_latency_ms", sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_recognition_sessions"),
    )
    op.create_index("ix_recognition_sessions_status", "recognition_sessions", ["status"])
    op.create_index("ix_recognition_sessions_started_at", "recognition_sessions", ["started_at"])

    op.create_table(
        "recognition_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.String(36), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("predicted_label", sa.String(120), nullable=True),
        sa.Column("selected_label", sa.String(120), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("inference_ms", sa.Float(), nullable=False),
        sa.Column("accepted", sa.Boolean(), nullable=False),
        sa.Column("corrected", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["recognition_sessions.id"],
            ondelete="CASCADE",
            name="fk_recognition_events_session_id_recognition_sessions",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_recognition_events"),
    )
    op.create_index("ix_recognition_events_session_id", "recognition_events", ["session_id"])

    op.create_table(
        "lessons",
        sa.Column("id", sa.String(64), nullable=False),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("material_status", sa.String(64), nullable=False),
        sa.Column("exercises_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_lessons"),
    )
    op.create_index("ix_lessons_level", "lessons", ["level"])

    op.create_table(
        "lesson_attempts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("lesson_id", sa.String(64), nullable=True),
        sa.Column("gesture_id", sa.String(120), nullable=False),
        sa.Column("mode", sa.String(20), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("duration_seconds", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["lesson_id"], ["lessons.id"], ondelete="SET NULL", name="fk_lesson_attempts_lesson_id_lessons"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_lesson_attempts"),
    )
    op.create_index("ix_lesson_attempts_lesson_id", "lesson_attempts", ["lesson_id"])
    op.create_index("ix_lesson_attempts_gesture_id", "lesson_attempts", ["gesture_id"])
    op.create_index("ix_lesson_attempts_created_at", "lesson_attempts", ["created_at"])

    op.create_table(
        "srs_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("gesture_id", sa.String(120), nullable=False),
        sa.Column("repetitions", sa.Integer(), nullable=False),
        sa.Column("interval_days", sa.Integer(), nullable=False),
        sa.Column("easiness", sa.Float(), nullable=False),
        sa.Column("due_date", sa.Date(), nullable=False),
        sa.Column("last_quality", sa.Integer(), nullable=True),
        sa.Column("last_reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_srs_items"),
        sa.UniqueConstraint("gesture_id", name="uq_srs_items_gesture_id"),
    )
    op.create_index("ix_srs_items_due_date", "srs_items", ["due_date"])

    op.create_table(
        "training_classes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(80), nullable=False),
        sa.Column("normalized_name", sa.String(80), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("is_negative", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_training_classes"),
        sa.UniqueConstraint("normalized_name", name="uq_training_classes_normalized_name"),
    )
    op.create_table(
        "training_sessions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("class_id", sa.Integer(), nullable=False),
        sa.Column("note", sa.String(200), nullable=False),
        sa.Column("image_consent", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["class_id"],
            ["training_classes.id"],
            ondelete="CASCADE",
            name="fk_training_sessions_class_id_training_classes",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_training_sessions"),
    )
    op.create_index("ix_training_sessions_class_id", "training_sessions", ["class_id"])

    op.create_table(
        "training_samples",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("features_json", sa.Text(), nullable=False),
        sa.Column("frame_count", sa.Integer(), nullable=False),
        sa.Column("image_path", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["training_sessions.id"],
            ondelete="CASCADE",
            name="fk_training_samples_session_id_training_sessions",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_training_samples"),
    )
    op.create_index("ix_training_samples_session_id", "training_samples", ["session_id"])
    op.create_index("ix_training_samples_kind", "training_samples", ["kind"])

    op.create_table(
        "training_jobs",
        sa.Column("id", sa.String(36), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("sample_kind", sa.String(16), nullable=False),
        sa.Column("activate_on_success", sa.Boolean(), nullable=False),
        sa.Column("dataset_version", sa.String(64), nullable=True),
        sa.Column("model_id", sa.String(64), nullable=True),
        sa.Column("metrics_json", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["model_id"],
            ["model_registry.id"],
            ondelete="SET NULL",
            name="fk_training_jobs_model_id_model_registry",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_training_jobs"),
    )
    op.create_index("ix_training_jobs_status", "training_jobs", ["status"])

    if legacy:
        _migrate_legacy(bind, existing)


def _parse_dt(value: object) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, str) and value:
        parsed = datetime.fromisoformat(value)
        return parsed.astimezone(UTC).replace(tzinfo=None) if parsed.tzinfo else parsed
    return datetime.now(UTC).replace(tzinfo=None)


def _migrate_legacy(bind: sa.Connection, existing: set[str]) -> None:
    """Copy rows from the pre-Alembic schema; legacy datetimes were naive UTC already."""
    rows = bind.execute(sa.text("SELECT id, name, description, is_unknown, created_at FROM gesture_classes"))
    seen: set[str] = set()
    for row in rows.mappings():
        normalized = str(row["name"]).strip().casefold()
        if normalized in seen:
            normalized = f"{normalized}-{row['id']}"
        seen.add(normalized)
        bind.execute(
            sa.text(
                "INSERT INTO training_classes (id, name, normalized_name, description, is_negative, created_at) "
                "VALUES (:id, :name, :normalized, :description, :neg, :created)"
            ),
            {
                "id": row["id"],
                "name": row["name"],
                "normalized": normalized,
                "description": row["description"] or "",
                "neg": bool(row["is_unknown"]),
                "created": _parse_dt(row["created_at"]),
            },
        )
    bind.execute(
        sa.text(
            "INSERT INTO training_sessions (id, class_id, note, image_consent, created_at) "
            "SELECT id, gesture_class_id, COALESCE(note, ''), 0, created_at FROM capture_sessions"
        )
    )
    bind.execute(
        sa.text(
            "INSERT INTO training_samples (id, session_id, kind, features_json, frame_count, image_path, created_at) "
            "SELECT id, session_id, 'static', features_json, 1, NULL, created_at FROM legacy_training_samples"
        )
    )
    if "attempts" in existing:
        bind.execute(
            sa.text(
                "INSERT INTO lesson_attempts (lesson_id, gesture_id, mode, success, confidence, duration_seconds, "
                "created_at) SELECT NULL, gesture_name, mode, success, confidence, duration_seconds, created_at "
                "FROM attempts"
            )
        )
    if "review_states" in existing:
        now = datetime.now(UTC).replace(tzinfo=None)
        bind.execute(
            sa.text(
                "INSERT INTO srs_items (gesture_id, repetitions, interval_days, easiness, due_date, last_quality, "
                "last_reviewed_at, created_at, updated_at) SELECT gesture_name, repetitions, interval_days, easiness, "
                "due_date, last_quality, NULL, :now, :now FROM review_states"
            ),
            {"now": now},
        )
    op.drop_table("legacy_training_samples")
    for table in ("capture_sessions", "gesture_classes", "attempts", "review_states"):
        if table in existing:
            op.drop_table(table)


def downgrade() -> None:
    for index, table in (
        ("ix_training_jobs_status", "training_jobs"),
        ("ix_training_samples_kind", "training_samples"),
        ("ix_training_samples_session_id", "training_samples"),
        ("ix_training_sessions_class_id", "training_sessions"),
        ("ix_srs_items_due_date", "srs_items"),
        ("ix_lesson_attempts_created_at", "lesson_attempts"),
        ("ix_lesson_attempts_gesture_id", "lesson_attempts"),
        ("ix_lesson_attempts_lesson_id", "lesson_attempts"),
        ("ix_lessons_level", "lessons"),
        ("ix_recognition_events_session_id", "recognition_events"),
        ("ix_recognition_sessions_started_at", "recognition_sessions"),
        ("ix_recognition_sessions_status", "recognition_sessions"),
        ("ix_model_registry_type", "model_registry"),
    ):
        op.drop_index(index, table_name=table)
    for table in (
        "training_jobs",
        "training_samples",
        "training_sessions",
        "training_classes",
        "srs_items",
        "lesson_attempts",
        "lessons",
        "recognition_events",
        "recognition_sessions",
        "model_registry",
        "settings",
    ):
        op.drop_table(table)
