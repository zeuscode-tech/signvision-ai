"""Checks against the real built-in models (skipped until setup.sh has downloaded them)."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from app.recognition.model_loader import load_model
from tests.conftest import GRAY, jpeg, make_settings, start_client

MODELS = Path(__file__).resolve().parents[3] / "models"
HAS_EASY_SIGN = (MODELS / "easy-sign-s3d" / "s3d.onnx").is_file()
HAS_BUKVA = (MODELS / "bukva" / "bukva_mobilenetv2_tsm.onnx").is_file()


@pytest.mark.skipif(not HAS_EASY_SIGN, reason="easy-sign-s3d weights not downloaded")
def test_easy_sign_manifest_validates_with_test_inference() -> None:
    model = load_model(MODELS / "easy-sign-s3d" / "manifest.json", MODELS, ["CPUExecutionProvider"])
    assert len(model.labels) == 1599 and model.labels[model.no_event_index or 0] == "no"
    assert model.validation is not None and model.validation.valid


@pytest.mark.skipif(not HAS_EASY_SIGN, reason="easy-sign-s3d weights not downloaded")
def test_medical_pack_matches_easy_sign_labels() -> None:
    from app.schemas.domain_packs import DomainPackIn
    from app.services.domain_pack_service import missing_labels

    model = load_model(MODELS / "easy-sign-s3d" / "manifest.json", MODELS, ["CPUExecutionProvider"])
    pack = DomainPackIn.model_validate_json((MODELS.parent / "domain-packs" / "medical.json").read_text())
    assert missing_labels(pack, model) == []


@pytest.mark.skipif(not HAS_BUKVA, reason="Bukva weights missing")
def test_bukva_rejects_an_empty_scene(copy_real_models: Path, workspace: Path) -> None:
    client = start_client(
        make_settings(workspace, model_root=copy_real_models, active_model_id="bukva-mobilenetv2-tsm")
    )
    try:
        assert client.get("/api/v1/health").json()["model_status"] == "ready"
        frame = "data:image/jpeg;base64," + base64.b64encode(jpeg(GRAY, (320, 240))).decode()
        result = client.post("/api/bukva/predict", json={"frames": [frame] * 8}).json()
        assert len(result["top3"]) == 3 and result["label"] is None and not result["accepted"]
    finally:
        client.__exit__(None, None, None)
