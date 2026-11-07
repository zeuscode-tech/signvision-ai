"""Application settings.

Values come from environment variables (or an optional `.env` in the project
root) and default to safe local values. Relative paths are resolved against the
project root, so the backend behaves the same regardless of the working
directory it was started from.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = PROJECT_ROOT / "backend"

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def resolve_path(value: Path) -> Path:
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def is_local_origin(origin: str) -> bool:
    parts = urlsplit(origin)
    return parts.scheme in {"http", "https"} and (parts.hostname or "") in LOOPBACK_HOSTS


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)
    database_url: str = "sqlite:///./data/signvision.db"
    data_dir: Path = Path("./data")
    log_dir: Path = Path("./data/logs")

    model_root: Path = Path("./models")
    active_model_id: str = "easy-sign-s3d"
    domain_pack_root: Path = Path("./domain-packs")
    onnx_providers: str = "CPUExecutionProvider"

    hand_detector: Literal["mediapipe", "none"] = "mediapipe"
    hand_landmarker_path: Path = Path("./frontend/public/models/hand_landmarker.task")

    max_frame_size_bytes: int = Field(default=1_000_000, ge=1_024, le=10_000_000)
    max_websocket_fps: float = Field(default=12, gt=0, le=60)
    frame_buffer_size: int = Field(default=32, ge=1, le=256)
    max_websocket_connections: int = Field(default=4, ge=1, le=32)
    inference_workers: int = Field(default=2, ge=1, le=16)

    default_confidence_threshold: float = Field(default=0.65, ge=0, le=1)
    default_stability_count: int = Field(default=2, ge=1, le=20)
    default_cooldown_ms: int = Field(default=1200, ge=0, le=60_000)
    pause_ms: int = Field(default=600, ge=0, le=10_000)

    save_raw_frames: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    allowed_origins: str = "http://127.0.0.1:5173,http://localhost:5173"

    max_import_size_bytes: int = Field(default=100_000_000, ge=1_000)
    max_import_uncompressed_bytes: int = Field(default=400_000_000, ge=1_000)

    @field_validator("app_host")
    @classmethod
    def host_must_be_loopback(cls, value: str) -> str:
        if value not in LOOPBACK_HOSTS:
            raise ValueError("APP_HOST должен быть loopback-адресом (127.0.0.1, ::1 или localhost)")
        return value

    @field_validator("allowed_origins")
    @classmethod
    def origins_must_be_local(cls, value: str) -> str:
        for origin in (item.strip() for item in value.split(",")):
            if origin and not is_local_origin(origin):
                raise ValueError(f"ALLOWED_ORIGINS допускает только локальный frontend, получено: {origin}")
        return value

    @field_validator("database_url")
    @classmethod
    def sqlite_only(cls, value: str) -> str:
        if not value.startswith("sqlite://"):
            raise ValueError("Поддерживается только локальная SQLite (DATABASE_URL=sqlite:///...)")
        return value

    @model_validator(mode="after")
    def resolve_paths(self) -> Settings:
        for name in ("data_dir", "log_dir", "model_root", "domain_pack_root", "hand_landmarker_path"):
            setattr(self, name, resolve_path(getattr(self, name)))
        return self

    @property
    def origins(self) -> list[str]:
        return [item.strip() for item in self.allowed_origins.split(",") if item.strip()]

    @property
    def providers(self) -> list[str]:
        return [item.strip() for item in self.onnx_providers.split(",") if item.strip()]

    @property
    def resolved_database_url(self) -> str:
        """Resolve a relative SQLite file path against the project root."""
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            return self.database_url  # "sqlite://" = in-memory
        raw = self.database_url[len(prefix) :]
        if raw in {"", ":memory:"} or raw.startswith("/"):
            return self.database_url
        return prefix + str(resolve_path(Path(raw)))

    @property
    def database_file(self) -> Path | None:
        url = self.resolved_database_url
        raw = url[len("sqlite:///") :] if url.startswith("sqlite:///") else ""
        return Path(raw) if raw and raw != ":memory:" else None

    @property
    def user_model_root(self) -> Path:
        return self.model_root / "user"

    @property
    def tmp_dir(self) -> Path:
        return self.data_dir / "tmp"

    @property
    def training_image_dir(self) -> Path:
        return self.data_dir / "training_images"

    @property
    def raw_frame_dir(self) -> Path:
        return self.data_dir / "raw_frames"


def ensure_directories(settings: Settings) -> None:
    for path in (
        settings.data_dir,
        settings.log_dir,
        settings.model_root,
        settings.user_model_root,
        settings.domain_pack_root,
        settings.tmp_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    db_file = settings.database_file
    if db_file is not None:
        db_file.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
