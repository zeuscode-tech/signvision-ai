"""Service container stored on `app.state` and FastAPI dependencies."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from fastapi import Request
from sqlalchemy.orm import Session
from starlette.requests import HTTPConnection

from .core.config import Settings
from .core.security import LocalSigner
from .database import Database
from .recognition.hand_detection import HandDetectorFactory
from .recognition.inference import InferenceExecutor
from .services.data_service import DataService
from .services.domain_pack_service import DomainPackService
from .services.model_service import ModelService
from .services.training_service import TrainingService
from .websocket.connection_manager import ConnectionManager


@dataclass
class Container:
    settings: Settings
    database: Database
    signer: LocalSigner
    models: ModelService
    packs: DomainPackService
    training: TrainingService
    data: DataService
    inference: InferenceExecutor
    preprocess: InferenceExecutor
    hands: HandDetectorFactory
    connections: ConnectionManager

    @classmethod
    def build(cls, settings: Settings) -> Container:
        database = Database(settings.resolved_database_url)
        signer = LocalSigner(settings.data_dir / "secret.key")
        training = TrainingService(settings, database, signer)
        return cls(
            settings=settings,
            database=database,
            signer=signer,
            models=ModelService(settings, database),
            packs=DomainPackService(settings),
            training=training,
            data=DataService(settings, training),
            inference=InferenceExecutor(settings.inference_workers),
            preprocess=InferenceExecutor(2),
            hands=HandDetectorFactory(settings.hand_detector, settings.hand_landmarker_path),
            connections=ConnectionManager(settings.max_websocket_connections),
        )

    def shutdown(self) -> None:
        self.training.shutdown()
        self.inference.shutdown()
        self.preprocess.shutdown()
        self.database.dispose()


def get_container(connection: HTTPConnection) -> Container:
    container: Container = connection.app.state.container
    return container


def get_db(request: Request) -> Iterator[Session]:
    with get_container(request).database.session() as db:
        yield db
