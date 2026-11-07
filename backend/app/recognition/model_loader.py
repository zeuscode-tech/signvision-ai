"""Load and validate ONNX recognition models described by a manifest.

Validation covers: manifest schema, paths staying inside MODEL_ROOT, file
existence, ONNX graph correctness, input shape, label count vs model output and
an actual test inference. The ONNX session is created once per load.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from pydantic import ValidationError

from ..core.exceptions import ErrorCode
from ..core.security import UnsafePathError, ensure_within, safe_relative_path, sha256_file
from ..core.timeutils import utcnow
from ..schemas.manifest import ModelManifest

logger = logging.getLogger(__name__)

MAX_MODEL_BYTES = 1_000_000_000
MAX_LABELS_BYTES = 5_000_000


def normalize_label(value: str) -> str:
    return value.strip().casefold()


@dataclass
class ValidationCheck:
    name: str
    ok: bool
    message: str


@dataclass
class ValidationReport:
    model_id: str | None
    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return bool(self.checks) and all(check.ok for check in self.checks)

    def passed(self, name: str, message: str = "ok") -> None:
        self.checks.append(ValidationCheck(name, True, message))

    def failed(self, name: str, message: str) -> None:
        self.checks.append(ValidationCheck(name, False, message))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "valid": self.valid,
            "checks": [{"name": c.name, "ok": c.ok, "message": c.message} for c in self.checks],
        }


class ModelLoadError(Exception):
    def __init__(self, code: ErrorCode, message: str, report: ValidationReport) -> None:
        self.code = code
        self.message = message
        self.report = report
        super().__init__(message)


@dataclass
class LoadedModel:
    manifest: ModelManifest
    session: ort.InferenceSession
    input_name: str
    labels: list[str]
    providers: list[str]
    manifest_rel_path: str
    model_rel_path: str
    labels_rel_path: str
    sha256: str
    loaded_at: Any
    validation: ValidationReport | None = None
    label_index: dict[str, int] = field(default_factory=dict)
    no_event_index: int | None = None

    def __post_init__(self) -> None:
        self.label_index = {normalize_label(label): index for index, label in enumerate(self.labels)}
        if self.manifest.no_event_label is not None:
            self.no_event_index = self.label_index.get(normalize_label(self.manifest.no_event_label))

    @property
    def id(self) -> str:
        return self.manifest.id

    def run(self, tensor: np.ndarray) -> np.ndarray:
        """Run the ONNX graph; returns the flat output vector. Thread-safe (ORT sessions are)."""
        output = self.session.run(None, {self.input_name: tensor})[0]
        return np.asarray(output, dtype=np.float32).reshape(-1)

    def probabilities(self, tensor: np.ndarray) -> np.ndarray:
        return to_probabilities(self.run(tensor), self.manifest.output)


def to_probabilities(raw: np.ndarray, kind: str) -> np.ndarray:
    """Convert the model's real output to probabilities according to the manifest."""
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if kind == "probabilities":
        return values.astype(np.float32)
    shifted = values - values.max()
    exp = np.exp(shifted)
    return (exp / exp.sum()).astype(np.float32)


def resolve_inside(model_root: Path, base_dir: Path, value: str) -> Path:
    relative = safe_relative_path(value)
    return ensure_within(model_root, base_dir / Path(*relative.parts))


def relative_to_root(model_root: Path, path: Path) -> str:
    return path.resolve().relative_to(model_root.resolve()).as_posix()


def read_manifest(manifest_path: Path) -> ModelManifest:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return ModelManifest.model_validate(raw)


def _read_labels(path: Path) -> list[str]:
    if path.stat().st_size > MAX_LABELS_BYTES:
        raise ValueError("Файл labels слишком большой")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("labels должен быть непустым JSON-массивом строк")
    if not all(isinstance(item, str) and item.strip() for item in data):
        raise ValueError("Каждый label должен быть непустой строкой")
    normalized = [normalize_label(item) for item in data]
    if len(set(normalized)) != len(normalized):
        raise ValueError("labels содержит повторяющиеся значения")
    return [str(item) for item in data]


def _choose_providers(requested: list[str]) -> list[str]:
    available = set(ort.get_available_providers())
    chosen = [provider for provider in requested if provider in available]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


def _fail(report: ValidationReport, code: ErrorCode, name: str, message: str) -> ModelLoadError:
    report.failed(name, message)
    return ModelLoadError(code, message, report)


def load_model(manifest_path: Path, model_root: Path, default_providers: list[str]) -> LoadedModel:
    """Fully validate and load a model. Raises ModelLoadError with a report on any failure."""
    report = ValidationReport(model_id=None)
    model_root = model_root.resolve()
    try:
        manifest_path = ensure_within(model_root, manifest_path)
    except UnsafePathError as exc:
        raise _fail(report, ErrorCode.MODEL_INVALID, "paths", str(exc)) from exc
    if not manifest_path.is_file():
        raise _fail(report, ErrorCode.MODEL_NOT_FOUND, "manifest", "Manifest модели не найден")
    try:
        manifest = read_manifest(manifest_path)
    except (ValueError, ValidationError) as exc:
        message = "Manifest не прошёл проверку схемы"
        if isinstance(exc, ValidationError):
            message += ": " + "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()[:5]
            )
        raise _fail(report, ErrorCode.MODEL_INVALID, "manifest", message) from exc
    report.model_id = manifest.id
    report.passed("manifest")

    try:
        model_file = resolve_inside(model_root, manifest_path.parent, manifest.model_path)
        labels_file = resolve_inside(model_root, manifest_path.parent, manifest.labels_path)
    except UnsafePathError as exc:
        raise _fail(report, ErrorCode.MODEL_INVALID, "paths", f"{exc}: model_path/labels_path") from exc
    report.passed("paths", "Файлы находятся внутри каталога моделей")

    missing = [name for name, path in (("model", model_file), ("labels", labels_file)) if not path.is_file()]
    if missing:
        raise _fail(
            report,
            ErrorCode.MODEL_NOT_FOUND,
            "files",
            "Не найдены файлы: " + ", ".join(missing) + ". Запустите setup.sh, чтобы скачать модель.",
        )
    if model_file.stat().st_size > MAX_MODEL_BYTES:
        raise _fail(report, ErrorCode.MODEL_INVALID, "files", "Файл модели превышает 1 ГБ")
    report.passed("files")

    try:
        onnx.checker.check_model(str(model_file))
    except Exception as exc:  # onnx raises several exception types
        raise _fail(
            report, ErrorCode.MODEL_INVALID, "onnx_graph", f"Некорректный ONNX-граф: {type(exc).__name__}"
        ) from exc
    report.passed("onnx_graph")

    try:
        labels = _read_labels(labels_file)
    except (ValueError, OSError) as exc:
        raise _fail(report, ErrorCode.MODEL_INVALID, "labels", str(exc)) from exc
    if manifest.no_event_label is not None and normalize_label(manifest.no_event_label) not in {
        normalize_label(label) for label in labels
    }:
        raise _fail(report, ErrorCode.MODEL_INVALID, "labels", "no_event_label отсутствует в labels")
    report.passed("labels", f"{len(labels)} классов")

    providers = _choose_providers(manifest.providers or default_providers)
    session_error: Exception | None = None
    try:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        session = ort.InferenceSession(str(model_file), sess_options=options, providers=providers)
    except Exception as exc:
        session_error = exc
        # Hardware providers can be available according to ONNX Runtime but still fail
        # while compiling a particular graph (for example CoreML in a restricted temp
        # directory). A demo must not lose the whole model when the portable CPU provider
        # can run it, so retry once on CPU before declaring the model invalid.
        if providers != ["CPUExecutionProvider"] and "CPUExecutionProvider" in providers:
            logger.warning(
                "Model %s failed with providers=%s (%s); retrying on CPU",
                manifest.id,
                ",".join(providers),
                type(exc).__name__,
            )
            try:
                options = ort.SessionOptions()
                options.log_severity_level = 3
                session = ort.InferenceSession(
                    str(model_file), sess_options=options, providers=["CPUExecutionProvider"]
                )
            except Exception as cpu_exc:
                raise _fail(
                    report,
                    ErrorCode.MODEL_INVALID,
                    "session",
                    "ONNX Runtime не смог загрузить модель ни через аппаратный provider, ни через CPU: "
                    f"{type(cpu_exc).__name__}",
                ) from cpu_exc
        else:
            raise _fail(
                report,
                ErrorCode.MODEL_INVALID,
                "session",
                f"ONNX Runtime не смог загрузить модель: {type(exc).__name__}",
            ) from exc
    provider_message = ", ".join(session.get_providers())
    if session_error is not None:
        provider_message += f" (fallback после {type(session_error).__name__})"
    report.passed("session", provider_message)

    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise _fail(report, ErrorCode.MODEL_INVALID, "input_shape", "Модель должна иметь ровно один вход")
    model_input = inputs[0]
    if model_input.type != "tensor(float)":
        raise _fail(
            report,
            ErrorCode.MODEL_INVALID,
            "input_shape",
            f"Ожидается float-вход, у модели {model_input.type}",
        )
    declared = list(model_input.shape)
    if len(declared) != len(manifest.input_shape):
        raise _fail(
            report,
            ErrorCode.MODEL_INVALID,
            "input_shape",
            f"Ранг входа модели {len(declared)} не совпадает с manifest",
        )
    for axis, (model_dim, manifest_dim) in enumerate(zip(declared, manifest.input_shape, strict=True)):
        if isinstance(model_dim, int) and model_dim != manifest_dim:
            raise _fail(
                report,
                ErrorCode.MODEL_INVALID,
                "input_shape",
                f"Ось {axis}: модель ожидает {model_dim}, manifest указывает {manifest_dim}",
            )
    report.passed("input_shape", str(manifest.input_shape))

    outputs = session.get_outputs()
    declared_out = list(outputs[0].shape)
    if declared_out and isinstance(declared_out[-1], int) and declared_out[-1] != len(labels):
        raise _fail(
            report,
            ErrorCode.MODEL_INVALID,
            "output_size",
            f"Выход модели: {declared_out[-1]} классов, labels: {len(labels)}",
        )

    loaded = LoadedModel(
        manifest=manifest,
        session=session,
        input_name=model_input.name,
        labels=labels,
        providers=session.get_providers(),
        manifest_rel_path=relative_to_root(model_root, manifest_path),
        model_rel_path=relative_to_root(model_root, model_file),
        labels_rel_path=relative_to_root(model_root, labels_file),
        sha256=sha256_file(model_file),
        loaded_at=utcnow(),
        validation=report,
    )
    started = time.perf_counter()
    try:
        raw = loaded.run(np.zeros(manifest.input_shape, dtype=np.float32))
    except Exception as exc:
        raise _fail(
            report,
            ErrorCode.MODEL_INVALID,
            "test_inference",
            f"Тестовый инференс завершился ошибкой: {type(exc).__name__}",
        ) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    if raw.size != len(labels):
        raise _fail(
            report,
            ErrorCode.MODEL_INVALID,
            "output_size",
            f"Выход модели: {raw.size} значений, labels: {len(labels)}",
        )
    report.passed("output_size", f"{raw.size} классов")
    if not np.isfinite(raw).all():
        raise _fail(report, ErrorCode.MODEL_INVALID, "test_inference", "Выход содержит NaN/Infinity")
    if manifest.output == "probabilities" and (raw.min() < -1e-4 or abs(float(raw.sum()) - 1.0) > 1e-2):
        raise _fail(
            report,
            ErrorCode.MODEL_INVALID,
            "test_inference",
            "manifest указывает output=probabilities, но выход модели не является распределением",
        )
    report.passed("test_inference", f"{elapsed_ms:.0f} мс")
    logger.info(
        "Model %s loaded (%d labels, providers=%s, test inference %.0f ms)",
        manifest.id,
        len(labels),
        ",".join(loaded.providers),
        elapsed_ms,
    )
    return loaded
