"""Local-only access rules, path safety and integrity signatures."""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import stat
import zipfile
from pathlib import Path, PurePosixPath

from .config import Settings

# "testclient" is the peer name used by Starlette's in-process test client.
LOOPBACK_CLIENTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


def is_loopback_client(host: str | None) -> bool:
    return host is not None and (host in LOOPBACK_CLIENTS or host.startswith("127."))


def is_origin_allowed(origin: str | None, settings: Settings) -> bool:
    """Browsers always send Origin for WebSocket handshakes, so it must be allow-listed.

    Native local tools (tests, CLI) may omit it; those are only accepted from loopback.
    """
    if origin is None:
        return True
    return origin in settings.origins


class UnsafePathError(ValueError):
    pass


def ensure_within(base: Path, candidate: Path) -> Path:
    """Resolve `candidate` (relative to `base` if not absolute) and require it to stay inside `base`."""
    base_resolved = base.resolve()
    target = (candidate if candidate.is_absolute() else base_resolved / candidate).resolve()
    if target != base_resolved and base_resolved not in target.parents:
        raise UnsafePathError("Путь выходит за пределы разрешённого каталога")
    return target


def safe_relative_path(value: str) -> PurePosixPath:
    """Validate a relative path from untrusted input (manifest, archive member)."""
    if not value or "\\" in value or "\x00" in value:
        raise UnsafePathError("Недопустимый путь")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"..", ""} for part in path.parts) or ":" in path.parts[0]:
        raise UnsafePathError("Недопустимый путь")
    return path


def is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class LocalSigner:
    """HMAC signatures for files produced by this installation.

    User models are pickled scikit-learn objects. Unpickling executes code, so a
    model file is loaded only when its signature was produced with this
    machine's secret key (i.e. the backend itself wrote it).
    """

    def __init__(self, key_path: Path) -> None:
        self._key = self._load_or_create(key_path)

    @staticmethod
    def _load_or_create(path: Path) -> bytes:
        if path.exists():
            data = path.read_bytes()
            if len(data) >= 32:
                return data
        path.parent.mkdir(parents=True, exist_ok=True)
        data = secrets.token_bytes(32)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        return data

    def sign_file(self, path: Path) -> str:
        mac = hmac.new(self._key, digestmod=hashlib.sha256)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                mac.update(chunk)
        return mac.hexdigest()

    def verify_file(self, path: Path, signature: str) -> bool:
        return hmac.compare_digest(self.sign_file(path), signature.strip())
