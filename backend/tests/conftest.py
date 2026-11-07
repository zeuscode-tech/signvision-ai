"""Shared fixtures: an isolated data/model/pack tree and a tiny deterministic ONNX model.

The tiny model averages each colour channel over the clip and maps it to
logits, so test frames have predictable outputs:
    gray/black → "no" (the model's no-event class)
    red        → "врач"     green → "голова"     blue → "помогать"
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import onnx
import pytest
from fastapi.testclient import TestClient
from onnx import TensorProto, helper

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.core.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402

TINY_LABELS = ["no", "врач", "голова", "помогать"]
TINY_FRAMES = 4
TINY_SIZE = 16


def build_tiny_onnx(path: Path, n_labels: int = len(TINY_LABELS), frames: int = TINY_FRAMES) -> None:
    weights = np.zeros((3, n_labels), dtype=np.float32)
    for channel in range(min(3, n_labels - 1)):
        weights[channel, channel + 1] = 9.0
    bias = np.zeros(n_labels, dtype=np.float32)
    bias[0] = 6.0  # gray frames (all channels 0.5) fall to the no-event class
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["input", "axes"], ["pooled"], keepdims=0),
            helper.make_node("MatMul", ["pooled", "weights"], ["scores"]),
            helper.make_node("Add", ["scores", "bias"], ["logits"]),
        ],
        "tiny_sign",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, frames, TINY_SIZE, TINY_SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, n_labels])],
        initializer=[
            helper.make_tensor("axes", TensorProto.INT64, [3], [2, 3, 4]),
            helper.make_tensor("weights", TensorProto.FLOAT, weights.shape, weights.flatten().tolist()),
            helper.make_tensor("bias", TensorProto.FLOAT, bias.shape, bias.tolist()),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def tiny_manifest(model_id: str = "tiny-sign", **overrides: object) -> dict[str, object]:
    manifest: dict[str, object] = {
        "id": model_id,
        "name": "Tiny test model",
        "version": "1.0",
        "type": "isolated_sign",
        "model_path": "model.onnx",
        "labels_path": "labels.json",
        "input_shape": [1, 3, TINY_FRAMES, TINY_SIZE, TINY_SIZE],
        "frame_count": TINY_FRAMES,
        "output": "logits",
        "no_event_label": "no",
        "confidence_threshold": 0.65,
        "stability_count": 2,
        "cooldown_ms": 0,
        "inference_stride": 1,
        "preprocessing": {
            "color": "RGB",
            "width": TINY_SIZE,
            "height": TINY_SIZE,
            "resize": "stretch",
            "crop": "none",
            "normalization": "scale_0_1",
            "layout": "NCTHW",
        },
    }
    manifest.update(overrides)
    return manifest


def write_model_dir(
    directory: Path,
    manifest: dict[str, object] | None = None,
    labels: list[str] | None = None,
    n_outputs: int | None = None,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    labels = labels or TINY_LABELS
    build_tiny_onnx(directory / "model.onnx", n_labels=n_outputs or len(labels))
    (directory / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps(manifest or tiny_manifest(), ensure_ascii=False), encoding="utf-8"
    )
    return directory


def jpeg(color: tuple[int, int, int] = (128, 128, 128), size: tuple[int, int] = (64, 48)) -> bytes:
    """Solid-colour JPEG; colour given as RGB."""
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image[:] = (color[2], color[1], color[0])
    ok, data = cv2.imencode(".jpg", image)
    assert ok
    return data.tobytes()


RED, GREEN, BLUE, GRAY = (255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 128)


def make_settings(root: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "data_dir": root / "data",
        "log_dir": root / "data" / "logs",
        "database_url": f"sqlite:///{root / 'data' / 'test.db'}",
        "model_root": root / "models",
        "domain_pack_root": root / "packs",
        "active_model_id": "tiny-sign",
        "hand_detector": "none",
        "max_websocket_fps": 60,
        "frame_buffer_size": 32,
        "pause_ms": 300,
        "log_level": "WARNING",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    write_model_dir(tmp_path / "models" / "tiny")
    packs = tmp_path / "packs"
    packs.mkdir()
    (packs / "clinic.json").write_text(
        json.dumps(
            {
                "id": "clinic",
                "name": "Клиника",
                "version": "1.0",
                "enabled_labels": ["ВРАЧ", "ГОЛОВА"],
                "phrase_templates": [{"tokens": ["ВРАЧ", "ГОЛОВА"], "result": "Тестовый шаблон"}],
                "quick_responses": ["Где болит?"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return tmp_path


def start_client(settings: Settings) -> TestClient:
    app = create_app(settings)
    client = TestClient(app, raise_server_exceptions=False)
    client.__enter__()
    app.state.container.models.wait_until_loaded(30)
    return client


@pytest.fixture
def settings(workspace: Path) -> Settings:
    return make_settings(workspace)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    test_client = start_client(settings)
    try:
        yield test_client
    finally:
        test_client.__exit__(None, None, None)


@pytest.fixture
def copy_real_models(tmp_path: Path) -> Path:
    """Copy the real built-in model folders if they are present (setup.sh downloads them)."""
    source = BACKEND.parent / "models"
    target = tmp_path / "real-models"
    for name in ("easy-sign-s3d", "bukva"):
        if (source / name / "manifest.json").exists():
            shutil.copytree(source / name, target / name)
    return target
