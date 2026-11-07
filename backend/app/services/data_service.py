"""Export, import and full deletion of user data.

Export: ZIP with manifest.json (format + version + sha256 of every member),
settings, progress and Training Studio metadata; user models optionally.

Import never executes archive content. The archive is size-checked, every
member path is validated (no traversal, no symlinks, allow-listed names only),
members are extracted into a temporary directory, checksums and schemas are
verified, and only then the data is replaced in one database transaction.
Model files are accepted only with a valid signature of this installation.
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import IO, Any

from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from .. import __version__
from ..core.config import Settings
from ..core.exceptions import AppError, ErrorCode
from ..core.security import UnsafePathError, is_zip_symlink, safe_relative_path, sha256_bytes, sha256_file
from ..core.timeutils import utcnow
from ..db_models import (
    LessonAttempt,
    ModelRegistryEntry,
    SrsItem,
    TrainingClass,
    TrainingSample,
    TrainingSession,
)
from ..repositories import lesson_repo, model_repo, session_repo, settings_repo, training_repo
from ..schemas.data import (
    EXPORT_FORMAT,
    EXPORT_FORMAT_VERSION,
    DeleteResult,
    ExportManifest,
    ImportResult,
    ProgressFile,
    SettingsRecord,
    TrainingFile,
    UserModelRecord,
)
from ..training.features import feature_size
from ..training.model_registry import METADATA_FILE, MODEL_FILE, SIGNATURE_FILE
from .training_service import MODEL_TYPES, TrainingService

logger = logging.getLogger(__name__)

DELETE_CONFIRMATION = "DELETE_ALL_USER_DATA"
DATA_FILES = ("settings.json", "progress.json", "training.json")
MODEL_MEMBER_FILES = (MODEL_FILE, METADATA_FILE, SIGNATURE_FILE)
MAX_MEMBER_BYTES = 200_000_000
MAX_COMPRESSION_RATIO = 200
MAX_MEMBERS = 3_000


def _dump(model: BaseModel) -> bytes:
    return model.model_dump_json(indent=2).encode("utf-8")


class DataService:
    def __init__(self, settings: Settings, training: TrainingService) -> None:
        self.settings = settings
        self.training = training

    # ---- export ------------------------------------------------------------------
    def export(self, db: Session, include_models: bool) -> bytes:
        prefs = settings_repo.get_or_create(db)
        settings_file = SettingsRecord(
            theme=prefs.theme,
            language=prefs.language,
            active_domain_pack_id=prefs.active_domain_pack_id,
            confidence_threshold=prefs.confidence_threshold,
            stability_count=prefs.stability_count,
            cooldown_ms=prefs.cooldown_ms,
            speech_enabled=prefs.speech_enabled,
            speech_rate=prefs.speech_rate,
            speech_voice=prefs.speech_voice,
        )
        progress_file = ProgressFile.model_validate(
            {
                "lesson_attempts": [
                    {
                        "lesson_id": a.lesson_id,
                        "gesture_id": a.gesture_id,
                        "mode": a.mode,
                        "success": a.success,
                        "confidence": a.confidence,
                        "duration_seconds": a.duration_seconds,
                        "created_at": a.created_at,
                    }
                    for a in lesson_repo.list_attempts(db)
                ],
                "srs_items": [
                    {
                        "gesture_id": s.gesture_id,
                        "repetitions": s.repetitions,
                        "interval_days": s.interval_days,
                        "easiness": s.easiness,
                        "due_date": s.due_date,
                        "last_quality": s.last_quality,
                        "last_reviewed_at": s.last_reviewed_at,
                    }
                    for s in lesson_repo.list_srs(db)
                ],
            }
        )
        models: list[dict[str, Any]] = []
        model_files: dict[str, Path] = {}
        if include_models:
            for entry in model_repo.list_models(db, types=list(MODEL_TYPES.values()), source="user"):
                signature = self.training.store.signature_of(entry.model_path)
                model_file = self.settings.model_root / entry.model_path
                if signature is None or not model_file.is_file() or entry.file_sha256 is None:
                    continue
                models.append(
                    {
                        "id": entry.id,
                        "name": entry.name,
                        "version": entry.version,
                        "type": entry.type,
                        "labels_count": entry.labels_count,
                        "metrics": json.loads(entry.metrics_json) if entry.metrics_json else None,
                        "preprocessing": json.loads(entry.preprocessing_json or "{}"),
                        "created_at": entry.created_at,
                        "file_sha256": entry.file_sha256,
                        "signature": signature,
                    }
                )
                for name in MODEL_MEMBER_FILES:
                    path = model_file.parent / name
                    if path.is_file():
                        model_files[f"models/{entry.id}/{name}"] = path
        training_file = TrainingFile.model_validate(
            {
                "classes": [
                    {
                        "id": c.id,
                        "name": c.name,
                        "description": c.description,
                        "is_negative": c.is_negative,
                        "created_at": c.created_at,
                    }
                    for c in training_repo.list_classes(db)
                ],
                "sessions": [
                    {
                        "id": s.id,
                        "class_id": s.class_id,
                        "note": s.note,
                        "image_consent": s.image_consent,
                        "created_at": s.created_at,
                    }
                    for s in training_repo.list_sessions(db)
                ],
                # Normalised landmark features only; images are never exported.
                "samples": [
                    {
                        "id": s.id,
                        "session_id": s.session_id,
                        "kind": s.kind,
                        "features": json.loads(s.features_json),
                        "frame_count": s.frame_count,
                        "created_at": s.created_at,
                    }
                    for s in training_repo.all_samples(db)
                ],
                "models": models,
            }
        )
        members = {
            "settings.json": _dump(settings_file),
            "progress.json": _dump(progress_file),
            "training.json": _dump(training_file),
        }
        hashes = {name: sha256_bytes(data) for name, data in members.items()}
        hashes.update({name: sha256_file(path) for name, path in model_files.items()})
        manifest = ExportManifest(
            format=EXPORT_FORMAT,
            format_version=EXPORT_FORMAT_VERSION,
            app_version=__version__,
            created_at=utcnow(),
            includes_models=bool(model_files),
            files=hashes,
        )
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", _dump(manifest))
            for name, data in members.items():
                archive.writestr(name, data)
            for name, path in model_files.items():
                archive.write(path, name)
        logger.info("User data exported (%d samples, %d models)", len(training_file.samples), len(models))
        return buffer.getvalue()

    # ---- import ------------------------------------------------------------------
    def save_upload(self, source: IO[bytes]) -> Path:
        """Stream the upload into a temp file, enforcing the size limit while reading."""
        self.settings.tmp_dir.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(dir=self.settings.tmp_dir, suffix=".zip", delete=False)  # noqa: SIM115
        total = 0
        try:
            with handle:
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    total += len(chunk)
                    if total > self.settings.max_import_size_bytes:
                        raise AppError(
                            ErrorCode.IMPORT_FORMAT_INVALID, "Файл импорта превышает допустимый размер"
                        )
                    handle.write(chunk)
        except Exception:
            Path(handle.name).unlink(missing_ok=True)
            raise
        return Path(handle.name)

    @staticmethod
    def _allowed_member(name: str) -> bool:
        if name in ("manifest.json", *DATA_FILES):
            return True
        parts = name.split("/")
        return len(parts) == 3 and parts[0] == "models" and parts[2] in MODEL_MEMBER_FILES

    def _extract(self, archive_path: Path, target: Path) -> dict[str, Path]:
        if not zipfile.is_zipfile(archive_path):
            raise AppError(ErrorCode.IMPORT_FORMAT_INVALID, "Файл не является ZIP-архивом экспорта")
        extracted: dict[str, Path] = {}
        total = 0
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_MEMBERS:
                raise AppError(ErrorCode.IMPORT_FORMAT_INVALID, "Слишком много файлов в архиве")
            for info in infos:
                if info.is_dir():
                    continue
                try:
                    relative = safe_relative_path(info.filename)
                except UnsafePathError as exc:
                    raise AppError(
                        ErrorCode.IMPORT_FORMAT_INVALID, "Архив содержит недопустимый путь"
                    ) from exc
                if is_zip_symlink(info) or info.flag_bits & 0x1:
                    raise AppError(
                        ErrorCode.IMPORT_FORMAT_INVALID, "Архив содержит ссылку или зашифрованный файл"
                    )
                if not self._allowed_member(relative.as_posix()):
                    raise AppError(
                        ErrorCode.IMPORT_FORMAT_INVALID, f"Неожиданный файл в архиве: {relative.name}"
                    )
                if relative.as_posix() in extracted:
                    raise AppError(ErrorCode.IMPORT_FORMAT_INVALID, "Повторяющиеся файлы в архиве")
                if info.file_size > MAX_MEMBER_BYTES or (
                    info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO
                ):
                    raise AppError(ErrorCode.IMPORT_FORMAT_INVALID, "Файл в архиве слишком большой")
                total += info.file_size
                if total > self.settings.max_import_uncompressed_bytes:
                    raise AppError(ErrorCode.IMPORT_FORMAT_INVALID, "Распакованный архив слишком большой")
                destination = target.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with archive.open(info) as source, destination.open("wb") as sink:
                    for chunk in iter(lambda src=source: src.read(1 << 20), b""):  # type: ignore[misc]
                        written += len(chunk)
                        if written > info.file_size:  # header lied about the size
                            raise AppError(ErrorCode.IMPORT_FORMAT_INVALID, "Повреждённый архив")
                        sink.write(chunk)
                extracted[relative.as_posix()] = destination
        return extracted

    @staticmethod
    def _load_json(path: Path, model: type[BaseModel]) -> Any:
        try:
            return model.model_validate_json(path.read_bytes())
        except (ValidationError, ValueError) as exc:
            raise AppError(
                ErrorCode.IMPORT_FORMAT_INVALID, f"Файл {path.name} не прошёл проверку схемы"
            ) from exc

    def import_archive(self, db: Session, archive_path: Path) -> ImportResult:
        workdir = Path(tempfile.mkdtemp(dir=self.settings.tmp_dir, prefix="import-"))
        try:
            files = self._extract(archive_path, workdir)
            if "manifest.json" not in files:
                raise AppError(ErrorCode.IMPORT_FORMAT_INVALID, "В архиве нет manifest.json")
            manifest: ExportManifest = self._load_json(files["manifest.json"], ExportManifest)
            if manifest.format_version != EXPORT_FORMAT_VERSION:
                raise AppError(
                    ErrorCode.IMPORT_FORMAT_INVALID,
                    f"Неподдерживаемая версия формата: {manifest.format_version}",
                )
            for name in DATA_FILES:
                if name not in files:
                    raise AppError(ErrorCode.IMPORT_FORMAT_INVALID, f"В архиве нет {name}")
            if set(files) - {"manifest.json"} != set(manifest.files):
                raise AppError(ErrorCode.IMPORT_FORMAT_INVALID, "Состав архива не совпадает с manifest")
            for name, expected in manifest.files.items():
                if sha256_file(files[name]) != expected:
                    raise AppError(ErrorCode.IMPORT_FORMAT_INVALID, f"Контрольная сумма {name} не совпадает")
            prefs: SettingsRecord = self._load_json(files["settings.json"], SettingsRecord)
            progress: ProgressFile = self._load_json(files["progress.json"], ProgressFile)
            training: TrainingFile = self._load_json(files["training.json"], TrainingFile)
            self._check_references(training)
            accepted_models, skipped = self._check_models(training.models, files)
            result = self._replace(db, prefs, progress, training, accepted_models)
            # Replaced data: previous Training Studio files are orphaned now.
            self.training.store.delete_all()
            shutil.rmtree(self.settings.training_image_dir, ignore_errors=True)
            self._install_models(accepted_models, files)
            result.skipped_models = skipped
            logger.info(
                "User data imported (%d samples, %d models)",
                result.imported_samples,
                len(result.imported_models),
            )
            return result
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @staticmethod
    def _check_references(training: TrainingFile) -> None:
        class_ids = {c.id for c in training.classes}
        session_ids = {s.id for s in training.sessions}
        if len(class_ids) != len(training.classes) or len(session_ids) != len(training.sessions):
            raise AppError(ErrorCode.IMPORT_FORMAT_INVALID, "Повторяющиеся идентификаторы")
        if len({c.name.casefold() for c in training.classes}) != len(training.classes):
            raise AppError(ErrorCode.IMPORT_FORMAT_INVALID, "Повторяющиеся названия классов")
        if any(s.class_id not in class_ids for s in training.sessions):
            raise AppError(ErrorCode.IMPORT_FORMAT_INVALID, "Сессия ссылается на несуществующий класс")
        for sample in training.samples:
            if sample.session_id not in session_ids:
                raise AppError(ErrorCode.IMPORT_FORMAT_INVALID, "Образец ссылается на несуществующую сессию")
            if len(sample.features) != feature_size(sample.kind):
                raise AppError(ErrorCode.IMPORT_FORMAT_INVALID, "Неверная длина вектора признаков")
            if any(value != value or abs(value) == float("inf") for value in sample.features):
                raise AppError(ErrorCode.IMPORT_FORMAT_INVALID, "Признаки содержат NaN/Infinity")

    def _check_models(
        self, models: list[UserModelRecord], files: dict[str, Path]
    ) -> tuple[list[UserModelRecord], list[dict[str, str]]]:
        accepted, skipped = [], []
        for record in models:
            model_file = files.get(f"models/{record.id}/{MODEL_FILE}")
            if model_file is None:
                skipped.append({"id": record.id, "reason": "file_missing"})
            elif sha256_file(model_file) != record.file_sha256:
                skipped.append({"id": record.id, "reason": "checksum_mismatch"})
            elif not self.training.store.signer.verify_file(model_file, record.signature):
                # Pickled models from another installation are never loaded: retrain from imported samples.
                skipped.append({"id": record.id, "reason": "signature_mismatch"})
            else:
                accepted.append(record)
        return accepted, skipped

    def _replace(
        self,
        db: Session,
        prefs: SettingsRecord,
        progress: ProgressFile,
        training: TrainingFile,
        models: list[UserModelRecord],
    ) -> ImportResult:
        try:
            self._wipe_user_rows(db)
            row = settings_repo.get_or_create(db)
            for key, value in prefs.model_dump().items():
                setattr(row, key, value)
            lesson_ids = {lesson.id for lesson in lesson_repo.list_lessons(db)}
            for attempt in progress.lesson_attempts:
                data = attempt.model_dump()
                if data["lesson_id"] not in lesson_ids:
                    data["lesson_id"] = None
                db.add(LessonAttempt(**data))
            for item in progress.srs_items:
                db.add(SrsItem(**item.model_dump(), created_at=utcnow(), updated_at=utcnow()))
            class_map: dict[int, TrainingClass] = {}
            for record in training.classes:
                gesture = TrainingClass(
                    name=record.name,
                    normalized_name=record.name.casefold(),
                    description=record.description,
                    is_negative=record.is_negative,
                    created_at=record.created_at,
                )
                db.add(gesture)
                class_map[record.id] = gesture
            db.flush()
            session_map: dict[int, TrainingSession] = {}
            for session_record in training.sessions:
                capture = TrainingSession(
                    class_id=class_map[session_record.class_id].id,
                    note=session_record.note,
                    image_consent=session_record.image_consent,
                    created_at=session_record.created_at,
                )
                db.add(capture)
                session_map[session_record.id] = capture
            db.flush()
            for sample in training.samples:
                db.add(
                    TrainingSample(
                        session_id=session_map[sample.session_id].id,
                        kind=sample.kind,
                        features_json=json.dumps(sample.features),
                        frame_count=sample.frame_count,
                        created_at=sample.created_at,
                    )
                )
            for model in models:
                db.add(
                    ModelRegistryEntry(
                        id=model.id,
                        name=model.name,
                        version=model.version,
                        type=model.type,
                        source="user",
                        manifest_path=None,
                        model_path=f"user/{model.id}/{MODEL_FILE}",
                        labels_path=None,
                        labels_count=model.labels_count,
                        preprocessing_json=json.dumps(model.preprocessing),
                        metrics_json=json.dumps(model.metrics, ensure_ascii=False) if model.metrics else None,
                        file_sha256=model.file_sha256,
                        status="ready",
                        is_active=False,
                        created_at=model.created_at,
                    )
                )
            db.commit()
        except Exception:
            db.rollback()
            raise
        return ImportResult(
            imported_attempts=len(progress.lesson_attempts),
            imported_srs_items=len(progress.srs_items),
            imported_classes=len(training.classes),
            imported_sessions=len(training.sessions),
            imported_samples=len(training.samples),
            imported_models=[m.id for m in models],
            skipped_models=[],
        )

    def _install_models(self, models: list[UserModelRecord], files: dict[str, Path]) -> None:
        for model in models:
            directory = self.training.store.directory(model.id)
            directory.mkdir(parents=True, exist_ok=True)
            for name in MODEL_MEMBER_FILES:
                source = files.get(f"models/{model.id}/{name}")
                if source is not None:
                    shutil.copyfile(source, directory / name)

    # ---- deletion ----------------------------------------------------------------
    def _wipe_user_rows(self, db: Session) -> dict[str, int]:
        removed: dict[str, int] = {}
        attempts, srs_items = lesson_repo.delete_progress(db)
        removed["lesson_attempts"], removed["srs_items"] = attempts, srs_items
        removed["recognition_sessions"] = session_repo.delete_all(db)
        removed.update(training_repo.delete_all(db))
        user_models = model_repo.list_models(db, types=list(MODEL_TYPES.values()), source="user")
        for entry in user_models:
            db.delete(entry)
        removed["user_models"] = len(user_models)
        db.flush()
        self.training.forget_cache()
        return removed

    def delete_all(self, db: Session, confirmation: str | None) -> DeleteResult:
        if confirmation != DELETE_CONFIRMATION:
            raise AppError(
                ErrorCode.CONFIRMATION_REQUIRED,
                f"Передайте confirm={DELETE_CONFIRMATION}, чтобы удалить все пользовательские данные",
            )
        images = training_repo.image_paths(db)
        removed = self._wipe_user_rows(db)
        db.commit()
        # Built-in ONNX models are never touched; only Training Studio versions and images.
        self.training.store.delete_all()
        shutil.rmtree(self.settings.training_image_dir, ignore_errors=True)
        shutil.rmtree(self.settings.raw_frame_dir, ignore_errors=True)
        removed["training_images"] = len(images)
        logger.info("All user data deleted")
        return DeleteResult(deleted=True, removed=removed)
