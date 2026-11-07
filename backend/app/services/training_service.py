"""Training Studio: classes, capture sessions, samples, background jobs, model versions."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from ..core.config import Settings
from ..core.exceptions import AppError, ErrorCode
from ..core.security import LocalSigner, ensure_within
from ..core.timeutils import utcnow
from ..database import Database
from ..db_models import ModelRegistryEntry, TrainingClass, TrainingJob, TrainingSample, TrainingSession
from ..recognition.preprocessing import InvalidFrameError, check_frame_bytes, decode_jpeg
from ..repositories import model_repo, training_repo
from ..schemas.training import (
    PredictIn,
    PredictOut,
    SampleCreate,
    SampleOut,
    TrainingClassCreate,
    TrainingClassOut,
    TrainingJobCreate,
    TrainingJobOut,
    TrainingModelOut,
    TrainingSessionCreate,
    TrainingSessionOut,
)
from ..training import features as feature_lib
from ..training.dataset import InsufficientDataError, build_dataset, check_sufficiency
from ..training.model_registry import ModelIntegrityError, UserModelStore
from ..training.trainer import predict_proba, train

logger = logging.getLogger(__name__)

MODEL_TYPES = {"static": "static_landmarks", "sequence": "sequence_landmarks"}
KIND_BY_TYPE = {value: key for key, value in MODEL_TYPES.items()}
MAX_IMAGE_BYTES = 2_000_000


def _json(value: str | None) -> Any:
    return json.loads(value) if value else None


class TrainingService:
    def __init__(self, settings: Settings, database: Database, signer: LocalSigner) -> None:
        self.settings = settings
        self.database = database
        self.store = UserModelStore(settings.model_root, settings.user_model_root, signer)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training")
        self._futures: dict[str, Future[None]] = {}
        self._model_cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def wait_for_job(self, job_id: str, timeout: float | None = None) -> None:
        future = self._futures.get(job_id)
        if future is not None:
            future.result(timeout)

    # ---- classes ---------------------------------------------------------------
    def list_classes(self, db: Session) -> list[TrainingClassOut]:
        counts = training_repo.class_counts(db)
        result = []
        for row in training_repo.list_classes(db):
            item = counts.get(row.id) or training_repo.ClassCounts()
            result.append(
                TrainingClassOut(
                    id=row.id,
                    name=row.name,
                    description=row.description,
                    is_negative=row.is_negative,
                    session_count=item.sessions,
                    sample_count=item.samples,
                    static_samples=item.static,
                    sequence_samples=item.sequence,
                    created_at=row.created_at,
                )
            )
        return result

    def create_class(self, db: Session, payload: TrainingClassCreate) -> TrainingClassOut:
        normalized = payload.name.casefold()
        if training_repo.find_class_by_name(db, normalized) is not None:
            raise AppError(ErrorCode.CONFLICT, "Класс с таким названием уже существует")
        row = TrainingClass(
            name=payload.name,
            normalized_name=normalized,
            description=payload.description,
            is_negative=payload.is_negative,
            created_at=utcnow(),
        )
        training_repo.add(db, row)
        db.commit()
        return TrainingClassOut(
            id=row.id,
            name=row.name,
            description=row.description,
            is_negative=row.is_negative,
            session_count=0,
            sample_count=0,
            static_samples=0,
            sequence_samples=0,
            created_at=row.created_at,
        )

    def delete_class(self, db: Session, class_id: int) -> None:
        row = training_repo.get_class(db, class_id)
        if row is None:
            raise AppError(ErrorCode.NOT_FOUND, "Класс не найден")
        images = [
            sample.image_path for session in row.sessions for sample in session.samples if sample.image_path
        ]
        training_repo.delete_row(db, row)
        db.commit()
        self._remove_images(images)

    # ---- sessions ----------------------------------------------------------------
    def create_session(self, db: Session, payload: TrainingSessionCreate) -> TrainingSessionOut:
        if training_repo.get_class(db, payload.class_id) is None:
            raise AppError(ErrorCode.NOT_FOUND, "Класс не найден")
        row = TrainingSession(
            class_id=payload.class_id,
            note=payload.note,
            image_consent=payload.image_consent,
            created_at=utcnow(),
        )
        training_repo.add(db, row)
        db.commit()
        return TrainingSessionOut(
            id=row.id,
            class_id=row.class_id,
            note=row.note,
            image_consent=row.image_consent,
            sample_count=0,
            created_at=row.created_at,
        )

    def list_sessions(self, db: Session, class_id: int | None) -> list[TrainingSessionOut]:
        counts = training_repo.session_sample_counts(db)
        return [
            TrainingSessionOut(
                id=row.id,
                class_id=row.class_id,
                note=row.note,
                image_consent=row.image_consent,
                sample_count=counts.get(row.id, 0),
                created_at=row.created_at,
            )
            for row in training_repo.list_sessions(db, class_id)
        ]

    # ---- samples -----------------------------------------------------------------
    def add_sample(self, db: Session, payload: SampleCreate) -> SampleOut:
        session = training_repo.get_session(db, payload.session_id)
        if session is None:
            raise AppError(ErrorCode.NOT_FOUND, "Сессия не найдена")
        try:
            vector = feature_lib.extract(payload.kind, payload.landmarks, payload.sequence)
        except feature_lib.FeatureError as exc:
            raise AppError(ErrorCode.VALIDATION_ERROR, str(exc)) from exc
        image_bytes: bytes | None = None
        if payload.image is not None:
            if not session.image_consent:
                raise AppError(
                    ErrorCode.VALIDATION_ERROR,
                    "Изображения сохраняются только после явного согласия (image_consent у сессии)",
                )
            image_bytes = self._decode_image(payload.image)
        row = TrainingSample(
            session_id=session.id,
            kind=payload.kind,
            features_json=json.dumps(vector.tolist()),
            frame_count=len(payload.sequence) if payload.sequence else 1,
            created_at=utcnow(),
        )
        training_repo.add(db, row)
        if image_bytes is not None:
            relative = f"training_images/{session.id}/{row.id}.jpg"
            target = ensure_within(self.settings.data_dir, self.settings.data_dir / relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(image_bytes)
            row.image_path = relative
        db.commit()
        return SampleOut(
            id=row.id,
            session_id=row.session_id,
            class_id=session.class_id,
            kind=row.kind,
            frame_count=row.frame_count,
            has_image=row.image_path is not None,
            created_at=row.created_at,
        )

    @staticmethod
    def _decode_image(value: str) -> bytes:
        encoded = value.split(",", 1)[1] if value.startswith("data:") else value
        try:
            data = base64.b64decode(encoded, validate=True)
            check_frame_bytes(data, MAX_IMAGE_BYTES)
            decode_jpeg(data)
        except (binascii.Error, ValueError, InvalidFrameError) as exc:
            raise AppError(ErrorCode.INVALID_FRAME, "Изображение должно быть JPEG до 2 МБ") from exc
        return data

    def list_samples(
        self,
        db: Session,
        *,
        session_id: int | None,
        class_id: int | None,
        kind: str | None,
        limit: int,
        offset: int,
    ) -> list[SampleOut]:
        return [
            SampleOut(
                id=s.id,
                session_id=s.session_id,
                class_id=cid,
                kind=s.kind,
                frame_count=s.frame_count,
                has_image=s.image_path is not None,
                created_at=s.created_at,
            )
            for s, cid in training_repo.list_samples(
                db, session_id=session_id, class_id=class_id, kind=kind, limit=limit, offset=offset
            )
        ]

    def delete_sample(self, db: Session, sample_id: int) -> None:
        row = training_repo.get_sample(db, sample_id)
        if row is None:
            raise AppError(ErrorCode.NOT_FOUND, "Образец не найден")
        image = row.image_path
        training_repo.delete_row(db, row)
        db.commit()
        self._remove_images([image] if image else [])

    def _remove_images(self, paths: list[str]) -> None:
        for relative in paths:
            try:
                target = ensure_within(self.settings.data_dir, self.settings.data_dir / relative)
                target.unlink(missing_ok=True)
            except (OSError, ValueError):
                logger.warning("Could not remove a training image")

    # ---- jobs --------------------------------------------------------------------
    def create_job(self, db: Session, payload: TrainingJobCreate) -> TrainingJobOut:
        dataset = build_dataset(payload.sample_kind, training_repo.dataset_rows(db, payload.sample_kind))
        try:
            check_sufficiency(dataset)  # reject immediately with a clear message
        except InsufficientDataError as exc:
            raise AppError(ErrorCode.TRAINING_DATA_INSUFFICIENT, str(exc), details=exc.details) from exc
        job = TrainingJob(
            id=str(uuid.uuid4()),
            status="queued",
            sample_kind=payload.sample_kind,
            activate_on_success=payload.activate_on_success,
            dataset_version=dataset.version,
            created_at=utcnow(),
        )
        training_repo.add(db, job)
        db.commit()
        self._futures[job.id] = self._executor.submit(self._run_job, job.id)
        logger.info("Training job %s queued (%s, dataset %s)", job.id, payload.sample_kind, dataset.version)
        return self.job_out(job)

    def get_job(self, db: Session, job_id: str) -> TrainingJobOut:
        job = training_repo.get_job(db, job_id)
        if job is None:
            raise AppError(ErrorCode.NOT_FOUND, "Задание обучения не найдено")
        return self.job_out(job)

    @staticmethod
    def job_out(job: TrainingJob) -> TrainingJobOut:
        return TrainingJobOut(
            id=job.id,
            status=job.status,
            sample_kind=job.sample_kind,
            activate_on_success=job.activate_on_success,
            dataset_version=job.dataset_version,
            model_id=job.model_id,
            metrics=_json(job.metrics_json),
            error_code=job.error_code,
            error_message=job.error_message,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
        )

    def _run_job(self, job_id: str) -> None:
        with self.database.session() as db:
            job = training_repo.get_job(db, job_id)
            if job is None:
                return
            job.status, job.started_at = "running", utcnow()
            db.commit()
            kind = job.sample_kind
            rows = training_repo.dataset_rows(db, kind)
        try:
            dataset = build_dataset(kind, rows)
            trained = train(dataset)
            model_id = self._save_version(kind, trained.estimator, trained.metrics, dataset.negative_labels)
        except InsufficientDataError as exc:
            self._fail(job_id, ErrorCode.TRAINING_DATA_INSUFFICIENT, str(exc))
            return
        except Exception:
            logger.exception("Training job %s failed", job_id)
            self._fail(
                job_id, ErrorCode.TRAINING_FAILED, "Обучение завершилось ошибкой; подробности в логе backend"
            )
            return
        with self.database.session() as db:
            job = training_repo.get_job(db, job_id)
            assert job is not None
            job.status, job.finished_at, job.model_id = "completed", utcnow(), model_id
            job.dataset_version = dataset.version
            job.metrics_json = json.dumps(trained.metrics, ensure_ascii=False)
            activate = job.activate_on_success
            db.commit()
        if activate:
            with self.database.session() as db:
                self.activate_model(db, model_id)
        logger.info(
            "Training job %s completed -> %s (macro F1 %.3f)", job_id, model_id, trained.metrics["macro_f1"]
        )

    def _fail(self, job_id: str, code: ErrorCode, message: str) -> None:
        with self.database.session() as db:
            job = training_repo.get_job(db, job_id)
            if job is not None:
                job.status, job.finished_at, job.error_code, job.error_message = (
                    "failed",
                    utcnow(),
                    str(code),
                    message,
                )
                db.commit()

    def _save_version(self, kind: str, estimator: Any, metrics: dict[str, Any], negatives: set[str]) -> str:
        model_type = MODEL_TYPES[kind]
        with self.database.session() as db:
            existing = model_repo.list_models(db, types=[model_type], source="user")
            version = 1 + max(
                (int(entry.version) for entry in existing if entry.version.isdigit()), default=0
            )
            model_id = f"user-{kind}-v{version}"
            while model_repo.get(db, model_id) is not None:
                version += 1
                model_id = f"user-{kind}-v{version}"
            artifact = {
                "model": estimator,
                "kind": kind,
                "labels": metrics["labels"],
                "negative_labels": sorted(negatives),
                "feature_version": feature_lib.FEATURE_VERSION,
                "threshold": self.settings.default_confidence_threshold,
            }
            stored = self.store.save(
                model_id,
                artifact,
                {"id": model_id, "version": version, "metrics": metrics, "created_at": utcnow().isoformat()},
            )
            model_repo.add(
                db,
                ModelRegistryEntry(
                    id=model_id,
                    name=f"Training Studio {kind} v{version}",
                    version=str(version),
                    type=model_type,
                    source="user",
                    manifest_path=None,
                    model_path=stored.relative_path,
                    labels_path=None,
                    labels_count=len(metrics["labels"]),
                    preprocessing_json=json.dumps(metrics["preprocessing"]),
                    metrics_json=json.dumps(metrics, ensure_ascii=False),
                    file_sha256=stored.sha256,
                    status="ready",
                    is_active=False,
                ),
            )
            db.commit()
        return model_id

    def mark_interrupted(self, db: Session) -> int:
        rows = training_repo.unfinished_jobs(db)
        for job in rows:
            job.status, job.finished_at = "failed", utcnow()
            job.error_code, job.error_message = (
                str(ErrorCode.TRAINING_FAILED),
                "Прервано перезапуском backend",
            )
        db.commit()
        return len(rows)

    # ---- model versions --------------------------------------------------------
    @staticmethod
    def model_out(entry: ModelRegistryEntry) -> TrainingModelOut:
        return TrainingModelOut(
            id=entry.id,
            name=entry.name,
            version=entry.version,
            type=entry.type,
            labels_count=entry.labels_count,
            metrics=_json(entry.metrics_json),
            is_active=entry.is_active,
            activated_at=entry.activated_at,
            created_at=entry.created_at,
        )

    def list_models(self, db: Session) -> list[TrainingModelOut]:
        return [
            self.model_out(e)
            for e in model_repo.list_models(db, types=list(MODEL_TYPES.values()), source="user")
        ]

    def _load_artifact(self, entry: ModelRegistryEntry) -> dict[str, Any]:
        with self._lock:
            cached = self._model_cache.get(entry.id)
        if cached is not None:
            return cached
        try:
            artifact = self.store.load(entry.model_path, entry.file_sha256)
        except ModelIntegrityError as exc:
            raise AppError(ErrorCode.MODEL_INVALID, str(exc)) from exc
        with self._lock:
            self._model_cache[entry.id] = artifact
        return artifact

    def activate_model(self, db: Session, model_id: str) -> TrainingModelOut:
        entry = model_repo.get(db, model_id)
        if entry is None or entry.type not in KIND_BY_TYPE:
            raise AppError(ErrorCode.MODEL_NOT_FOUND)
        artifact = self._load_artifact(entry)  # integrity check + load before switching
        probe = np.zeros(feature_lib.feature_size(KIND_BY_TYPE[entry.type]), dtype=np.float32)
        predict_proba(artifact["model"], probe)  # test inference
        model_repo.set_active(db, entry)
        db.commit()
        logger.info("Training Studio model %s activated", model_id)
        return self.model_out(entry)

    def rollback(self, db: Session, kind: str) -> TrainingModelOut:
        model_type = MODEL_TYPES[kind]
        current = model_repo.active(db, model_type)
        if current is None:
            raise AppError(ErrorCode.MODEL_NOT_READY, "Нет активной модели для отката")
        previous = model_repo.previous_active(db, model_type, current.id)
        if previous is None:
            raise AppError(ErrorCode.MODEL_NOT_FOUND, "Нет предыдущей версии для отката")
        return self.activate_model(db, previous.id)

    def active_model(self, db: Session, kind: str) -> tuple[ModelRegistryEntry, dict[str, Any]] | None:
        entry = model_repo.active(db, MODEL_TYPES[kind])
        if entry is None:
            return None
        return entry, self._load_artifact(entry)

    def predict(self, db: Session, payload: PredictIn) -> PredictOut:
        active = self.active_model(db, payload.kind)
        if active is None:
            raise AppError(
                ErrorCode.MODEL_NOT_READY, "Модель Training Studio ещё не обучена или не активирована"
            )
        entry, artifact = active
        try:
            vector = feature_lib.extract(payload.kind, payload.landmarks, payload.sequence)
        except feature_lib.FeatureError as exc:
            raise AppError(ErrorCode.VALIDATION_ERROR, str(exc)) from exc
        labels, probabilities = predict_proba(artifact["model"], vector)
        order = np.argsort(probabilities)[::-1]
        best = int(order[0])
        candidate, confidence = labels[best], float(probabilities[best])
        threshold = float(artifact.get("threshold", self.settings.default_confidence_threshold))
        is_negative = candidate in set(artifact.get("negative_labels", []))
        confident = confidence >= threshold
        accepted = confident and not is_negative
        state = "recognized" if accepted else "unknown" if confident and is_negative else "uncertain"
        top3 = (
            [{"label": labels[int(i)], "confidence": float(probabilities[int(i)])} for i in order[:3]]
            if confident
            else []
        )
        return PredictOut(
            state=state,
            label=candidate if accepted else None,
            candidate=candidate,
            confidence=confidence,
            accepted=accepted,
            is_negative=is_negative,
            top3=top3,
            model_id=entry.id,
        )

    def forget_cache(self) -> None:
        with self._lock:
            self._model_cache.clear()
