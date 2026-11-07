from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.security import LocalSigner, UnsafePathError, ensure_within, is_zip_symlink, safe_relative_path
from app.schemas.manifest import ModelManifest
from tests.conftest import tiny_manifest


@pytest.mark.parametrize("value", ["../x", "/etc/passwd", "a/../../b", "C:/x", "a\\b", ""])
def test_unsafe_relative_paths(value: str) -> None:
    with pytest.raises(UnsafePathError):
        safe_relative_path(value)


def test_ensure_within(tmp_path: Path) -> None:
    assert ensure_within(tmp_path, tmp_path / "a" / "b") == (tmp_path / "a" / "b").resolve()
    with pytest.raises(UnsafePathError):
        ensure_within(tmp_path / "a", tmp_path / "b")


def test_symlink_member_detection() -> None:
    info = zipfile.ZipInfo("link")
    info.external_attr = 0o120777 << 16
    assert is_zip_symlink(info)
    assert not is_zip_symlink(zipfile.ZipInfo("file"))


def test_signer_verifies_only_own_files(tmp_path: Path) -> None:
    file = tmp_path / "f.bin"
    file.write_bytes(b"data")
    signer = LocalSigner(tmp_path / "key")
    signature = signer.sign_file(file)
    assert signer.verify_file(file, signature)
    assert not LocalSigner(tmp_path / "other-key").verify_file(file, signature)
    assert (tmp_path / "key").stat().st_mode & 0o077 == 0


def test_manifest_shape_must_match_preprocessing() -> None:
    ModelManifest.model_validate(tiny_manifest())
    with pytest.raises(ValidationError, match="frame_count"):
        ModelManifest.model_validate(tiny_manifest(frame_count=8))
    with pytest.raises(ValidationError):
        ModelManifest.model_validate(tiny_manifest(unexpected=True))


def test_settings_refuse_non_local_host_and_origins() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, app_host="0.0.0.0")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Settings(_env_file=None, allowed_origins="https://example.com")  # type: ignore[call-arg]
    relative = Settings(_env_file=None, database_url="sqlite:///./data/x.db")  # type: ignore[call-arg]
    assert relative.resolved_database_url.startswith("sqlite:////") and relative.database_file is not None
