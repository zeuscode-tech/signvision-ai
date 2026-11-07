"""On-disk storage for Training Studio model versions.

Every version gets its own directory under MODEL_ROOT/user/<model_id>/ and is
never overwritten. The pickled estimator is loaded only after its SHA-256
matches the registry and its HMAC signature matches this installation's key.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from ..core.security import LocalSigner, ensure_within, sha256_file

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
SIGNATURE_FILE = "model.sig"


class ModelIntegrityError(ValueError):
    pass


@dataclass
class StoredModel:
    model_id: str
    relative_path: str
    sha256: str
    signature: str


class UserModelStore:
    def __init__(self, model_root: Path, user_root: Path, signer: LocalSigner) -> None:
        self.model_root = model_root
        self.user_root = user_root
        self.signer = signer

    def directory(self, model_id: str) -> Path:
        return ensure_within(self.user_root, self.user_root / model_id)

    def save(self, model_id: str, artifact: dict[str, Any], metadata: dict[str, Any]) -> StoredModel:
        directory = self.directory(model_id)
        if directory.exists():
            raise FileExistsError(f"Версия {model_id} уже существует")
        directory.mkdir(parents=True)
        model_file = directory / MODEL_FILE
        joblib.dump(artifact, model_file)
        signature = self.signer.sign_file(model_file)
        (directory / SIGNATURE_FILE).write_text(signature, encoding="utf-8")
        (directory / METADATA_FILE).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return StoredModel(
            model_id, model_file.relative_to(self.model_root).as_posix(), sha256_file(model_file), signature
        )

    def load(self, relative_path: str, expected_sha256: str | None) -> dict[str, Any]:
        model_file = ensure_within(self.user_root, self.model_root / relative_path)
        if not model_file.is_file():
            raise ModelIntegrityError("Файл модели не найден")
        if expected_sha256 and sha256_file(model_file) != expected_sha256:
            raise ModelIntegrityError("Контрольная сумма модели не совпадает с реестром")
        signature_file = model_file.parent / SIGNATURE_FILE
        if not signature_file.is_file() or not self.signer.verify_file(
            model_file, signature_file.read_text()
        ):
            raise ModelIntegrityError("Подпись модели недействительна: файл создан не этой установкой")
        artifact = joblib.load(model_file)
        if not isinstance(artifact, dict) or "model" not in artifact:
            raise ModelIntegrityError("Неожиданный формат файла модели")
        return artifact

    def signature_of(self, relative_path: str) -> str | None:
        signature_file = (self.model_root / relative_path).parent / SIGNATURE_FILE
        return signature_file.read_text().strip() if signature_file.is_file() else None

    def delete(self, model_id: str) -> None:
        directory = self.directory(model_id)
        if directory.is_dir():
            shutil.rmtree(directory)

    def delete_all(self) -> int:
        count = 0
        if self.user_root.is_dir():
            for child in self.user_root.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                    count += 1
        return count
