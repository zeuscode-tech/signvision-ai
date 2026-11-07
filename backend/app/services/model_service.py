"""Registry of ONNX recognition models and the runtime that holds loaded sessions.

Activation is atomic: the new model is fully loaded and must pass a test
inference before it replaces the active one. Until then the previous model
keeps serving; on failure it stays active. Open WebSocket connections keep the
model object they started with until they reconnect.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from ..core.config import Settings
from ..core.exceptions import AppError, ErrorCode
from ..core.security import UnsafePathError, ensure_within, safe_relative_path
from ..database import Database
from ..db_models import ModelRegistryEntry
from ..recognition.model_loader import (
    LoadedModel,
    ModelLoadError,
    ValidationReport,
    load_model,
    read_manifest,
)
from ..repositories import model_repo, settings_repo

logger = logging.getLogger(__name__)

ONNX_TYPE = "isolated_sign"
MAX_CACHED_MODELS = 3


class RuntimeStatus:
    READY = "ready"
    LOADING = "loading"
    UNAVAILABLE = "model_unavailable"
    INVALID = "invalid"
    NOT_LOADED = "not_loaded"


class ModelService:
    def __init__(self, settings: Settings, database: Database) -> None:
        self.settings = settings
        self.database = database
        self._cache: OrderedDict[str, LoadedModel] = OrderedDict()
        self._runtime: dict[str, str] = {}
        self._active_id: str | None = None
        self._lock = threading.RLock()
        self._load_lock = threading.Lock()
        self._startup_thread: threading.Thread | None = None

    # ---- discovery / startup -------------------------------------------------
    def discover_builtin(self) -> None:
        """Register every `MODEL_ROOT/<dir>/manifest.json` shipped with the app."""
        root = self.settings.model_root
        with self.database.session() as db:
            for manifest_path in sorted(root.glob("*/manifest.json")):
                if manifest_path.parent.name == "user":
                    continue
                try:
                    manifest = read_manifest(manifest_path)
                except Exception as exc:  # broken manifest must not stop startup
                    logger.warning(
                        "Skipping invalid manifest %s: %s", manifest_path.parent.name, type(exc).__name__
                    )
                    continue
                rel = manifest_path.relative_to(root).as_posix()
                entry = model_repo.get(db, manifest.id)
                base = manifest_path.parent.relative_to(root).as_posix()
                values = {
                    "name": manifest.name,
                    "version": manifest.version,
                    "type": manifest.type,
                    "manifest_path": rel,
                    "model_path": f"{base}/{manifest.model_path}",
                    "labels_path": f"{base}/{manifest.labels_path}",
                    "preprocessing_json": manifest.preprocessing.model_dump_json(),
                }
                if entry is None:
                    model_repo.add(
                        db,
                        ModelRegistryEntry(
                            id=manifest.id,
                            source="builtin",
                            labels_count=0,
                            status="unknown",
                            is_active=False,
                            **values,
                        ),
                    )
                elif entry.source in {"builtin", "registered"} and entry.manifest_path == rel:
                    for key, value in values.items():
                        setattr(entry, key, value)
            db.commit()

    def configured_active_id(self) -> str | None:
        with self.database.session() as db:
            entry = model_repo.active(db, ONNX_TYPE)
            if entry is not None:
                return entry.id
            row = settings_repo.get_or_create(db)
            db.commit()
            return row.active_model_id or self.settings.active_model_id or None

    def start_background_load(self) -> None:
        model_id = self.configured_active_id()
        with self._lock:
            self._active_id = model_id
            if model_id:
                self._runtime[model_id] = RuntimeStatus.LOADING
        if not model_id:
            return
        self._startup_thread = threading.Thread(
            target=self._startup_load, args=(model_id,), name="model-startup", daemon=True
        )
        self._startup_thread.start()

    def wait_until_loaded(self, timeout: float | None = None) -> None:
        if self._startup_thread is not None:
            self._startup_thread.join(timeout)

    def _startup_load(self, model_id: str) -> None:
        try:
            self.get(model_id)
        except AppError:
            return  # status already recorded
        with self.database.session() as db:
            entry = model_repo.get(db, model_id)
            if entry is not None and not entry.is_active:
                model_repo.set_active(db, entry)
            settings_repo.get_or_create(db).active_model_id = model_id
            db.commit()

    # ---- runtime ---------------------------------------------------------------
    @property
    def active_id(self) -> str | None:
        return self._active_id

    def active(self) -> LoadedModel | None:
        with self._lock:
            return self._cache.get(self._active_id) if self._active_id else None

    def runtime_status(self, model_id: str | None) -> str:
        if model_id is None:
            return RuntimeStatus.UNAVAILABLE
        with self._lock:
            if model_id in self._cache:
                return RuntimeStatus.READY
            return self._runtime.get(model_id, RuntimeStatus.NOT_LOADED)

    def active_status(self) -> str:
        return self.runtime_status(self._active_id)

    def _manifest_path(self, entry: ModelRegistryEntry) -> Path:
        if not entry.manifest_path:
            raise AppError(ErrorCode.MODEL_INVALID, "У модели нет manifest")
        return self.settings.model_root / entry.manifest_path

    def _load_fresh(self, model_id: str) -> LoadedModel:
        with self.database.session() as db:
            entry = model_repo.get(db, model_id)
            if entry is None:
                raise AppError(ErrorCode.MODEL_NOT_FOUND)
            if entry.type != ONNX_TYPE:
                raise AppError(
                    ErrorCode.MODEL_INVALID, "Это модель Training Studio, а не ONNX-модель распознавания"
                )
            manifest_path = self._manifest_path(entry)
        try:
            loaded = load_model(manifest_path, self.settings.model_root, self.settings.providers)
        except ModelLoadError as exc:
            self._record_status(model_id, exc.code, exc.message, exc.report)
            raise AppError(exc.code, exc.message, details=exc.report.to_dict()) from exc
        if loaded.id != model_id:
            raise AppError(ErrorCode.MODEL_INVALID, "id в manifest не совпадает с реестром")
        if loaded.manifest.frame_count > self.settings.frame_buffer_size:
            message = (
                f"Модели нужно {loaded.manifest.frame_count} кадров, "
                f"FRAME_BUFFER_SIZE={self.settings.frame_buffer_size}"
            )
            self._record_status(model_id, ErrorCode.MODEL_INVALID, message, None)
            raise AppError(ErrorCode.MODEL_INVALID, message)
        self._record_ready(loaded)
        return loaded

    def get(self, model_id: str) -> LoadedModel:
        with self._lock:
            cached = self._cache.get(model_id)
            if cached is not None:
                self._cache.move_to_end(model_id)
                return cached
        with self._load_lock:
            with self._lock:
                cached = self._cache.get(model_id)
                if cached is not None:
                    return cached
                self._runtime[model_id] = RuntimeStatus.LOADING
            try:
                loaded = self._load_fresh(model_id)
            except AppError as exc:
                with self._lock:
                    if self._runtime.get(model_id) == RuntimeStatus.LOADING:
                        self._runtime[model_id] = (
                            RuntimeStatus.UNAVAILABLE
                            if exc.code == ErrorCode.MODEL_NOT_FOUND
                            else RuntimeStatus.INVALID
                        )
                raise
            except Exception as exc:
                logger.exception("Unexpected error while loading model %s", model_id)
                with self._lock:
                    self._runtime[model_id] = RuntimeStatus.INVALID
                raise AppError(ErrorCode.MODEL_INVALID) from exc
            self._store(loaded)
            return loaded

    def _store(self, loaded: LoadedModel) -> None:
        with self._lock:
            self._cache[loaded.id] = loaded
            self._cache.move_to_end(loaded.id)
            self._runtime[loaded.id] = RuntimeStatus.READY
            while len(self._cache) > MAX_CACHED_MODELS:
                evict = next((key for key in self._cache if key != self._active_id), None)
                if evict is None:
                    break
                del self._cache[evict]
                self._runtime[evict] = RuntimeStatus.NOT_LOADED

    def _record_status(
        self, model_id: str, code: ErrorCode, message: str, report: ValidationReport | None
    ) -> None:
        status = "unavailable" if code == ErrorCode.MODEL_NOT_FOUND else "invalid"
        with self._lock:
            self._runtime[model_id] = (
                RuntimeStatus.UNAVAILABLE if status == "unavailable" else RuntimeStatus.INVALID
            )
        logger.warning("Model %s is %s: %s", model_id, status, message)
        with self.database.session() as db:
            entry = model_repo.get(db, model_id)
            if entry is not None:
                entry.status = status
                entry.status_message = message
                db.commit()

    def _record_ready(self, loaded: LoadedModel) -> None:
        with self.database.session() as db:
            entry = model_repo.get(db, loaded.id)
            if entry is not None:
                entry.status = "ready"
                entry.status_message = None
                entry.labels_count = len(loaded.labels)
                entry.file_sha256 = loaded.sha256
                entry.preprocessing_json = loaded.manifest.preprocessing.model_dump_json()
                db.commit()

    # ---- registry operations ---------------------------------------------------
    def register(self, manifest_rel: str) -> ModelRegistryEntry:
        try:
            relative = safe_relative_path(manifest_rel)
            manifest_path = ensure_within(
                self.settings.model_root, self.settings.model_root / Path(*relative.parts)
            )
        except UnsafePathError as exc:
            raise AppError(ErrorCode.MODEL_INVALID, str(exc)) from exc
        if (
            manifest_path.parent == self.settings.user_model_root
            or self.settings.user_model_root in manifest_path.parents
        ):
            raise AppError(ErrorCode.MODEL_INVALID, "Каталог user/ зарезервирован для Training Studio")
        try:
            loaded = load_model(manifest_path, self.settings.model_root, self.settings.providers)
        except ModelLoadError as exc:
            raise AppError(exc.code, exc.message, details=exc.report.to_dict()) from exc
        with self.database.session() as db:
            if model_repo.get(db, loaded.id) is not None:
                raise AppError(ErrorCode.CONFLICT, f"Модель {loaded.id} уже зарегистрирована")
            entry = model_repo.add(
                db,
                ModelRegistryEntry(
                    id=loaded.id,
                    name=loaded.manifest.name,
                    version=loaded.manifest.version,
                    type=ONNX_TYPE,
                    source="registered",
                    manifest_path=loaded.manifest_rel_path,
                    model_path=loaded.model_rel_path,
                    labels_path=loaded.labels_rel_path,
                    labels_count=len(loaded.labels),
                    preprocessing_json=loaded.manifest.preprocessing.model_dump_json(),
                    file_sha256=loaded.sha256,
                    status="ready",
                    is_active=False,
                ),
            )
            db.commit()
        self._store(loaded)
        logger.info("Model %s registered", loaded.id)
        return entry

    def validate(self, model_id: str) -> ValidationReport:
        with self.database.session() as db:
            entry = model_repo.get(db, model_id)
            if entry is None:
                raise AppError(ErrorCode.MODEL_NOT_FOUND)
            if entry.type != ONNX_TYPE:
                raise AppError(ErrorCode.MODEL_INVALID, "Проверка доступна только для ONNX-моделей")
            manifest_path = self._manifest_path(entry)
        try:
            loaded = load_model(manifest_path, self.settings.model_root, self.settings.providers)
        except ModelLoadError as exc:
            self._record_status(model_id, exc.code, exc.message, exc.report)
            if model_id != self._active_id:
                with self._lock:
                    self._cache.pop(model_id, None)
            return exc.report
        self._record_ready(loaded)
        assert loaded.validation is not None
        return loaded.validation

    def activate(self, model_id: str) -> LoadedModel:
        with self._load_lock:
            loaded = self._load_fresh(model_id)  # full validation + test inference first
        with self.database.session() as db:
            entry = model_repo.get(db, model_id)
            if entry is None:
                raise AppError(ErrorCode.MODEL_NOT_FOUND)
            model_repo.set_active(db, entry)
            settings_repo.get_or_create(db).active_model_id = model_id
            db.commit()
        with self._lock:
            previous = self._active_id
            self._active_id = model_id
        self._store(loaded)
        logger.info("Active recognition model switched %s -> %s", previous, model_id)
        return loaded

    def delete(self, model_id: str) -> None:
        with self.database.session() as db:
            entry = model_repo.get(db, model_id)
            if entry is None:
                raise AppError(ErrorCode.MODEL_NOT_FOUND)
            if entry.is_active or model_id == self._active_id:
                raise AppError(ErrorCode.MODEL_ACTIVE)
            if entry.type != ONNX_TYPE:
                raise AppError(
                    ErrorCode.VALIDATION_ERROR, "Модели Training Studio удаляются через /training/models"
                )
            model_repo.delete(db, entry)
            db.commit()
        with self._lock:
            self._cache.pop(model_id, None)
            self._runtime.pop(model_id, None)
        logger.info("Model %s removed from registry (files left untouched)", model_id)

    def labels(self, model_id: str) -> list[str] | None:
        with self._lock:
            cached = self._cache.get(model_id)
        return list(cached.labels) if cached else None

    def manifest_dict(self, model_id: str) -> dict[str, Any] | None:
        with self._lock:
            cached = self._cache.get(model_id)
        if cached is not None:
            return cached.manifest.model_dump(mode="json")
        with self.database.session() as db:
            entry = model_repo.get(db, model_id)
            if entry is None or not entry.manifest_path:
                return None
            path = self.settings.model_root / entry.manifest_path
        try:
            return dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return None
