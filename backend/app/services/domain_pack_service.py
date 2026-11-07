"""Domain packs: JSON files in DOMAIN_PACK_ROOT that restrict recognisable classes.

Rules: every label must exist in the model; an empty or corrupted pack cannot
be activated; phrase templates are written by people and need manual language
review (`review_status`); a list of glosses is never presented as a translation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from ..core.config import Settings
from ..core.exceptions import AppError, ErrorCode
from ..recognition.inference import ClassFilter
from ..recognition.model_loader import LoadedModel, normalize_label
from ..recognition.phrase_composer import PhraseComposer, PhraseTemplate
from ..schemas.domain_packs import Compatibility, DomainPackBroken, DomainPackIn, DomainPackOut
from ..schemas.manifest import MODEL_ID_PATTERN

logger = logging.getLogger(__name__)
MAX_PACK_BYTES = 2_000_000


@dataclass
class ResolvedPack:
    pack: DomainPackIn | None
    class_filter: ClassFilter
    composer: PhraseComposer


def missing_labels(pack: DomainPackIn, model: LoadedModel) -> list[str]:
    return [label for label in pack.enabled_labels if normalize_label(label) not in model.label_index]


class DomainPackService:
    def __init__(self, settings: Settings) -> None:
        self.root = settings.domain_pack_root

    def _path(self, pack_id: str) -> Path:
        if not re.fullmatch(MODEL_ID_PATTERN, pack_id):
            raise AppError(ErrorCode.DOMAIN_PACK_NOT_FOUND)
        return self.root / f"{pack_id}.json"

    def _read(self, path: Path) -> DomainPackIn:
        if path.stat().st_size > MAX_PACK_BYTES:
            raise ValueError("Файл словаря слишком большой")
        pack = DomainPackIn.model_validate(json.loads(path.read_text(encoding="utf-8")))
        if pack.id != path.stem:
            raise ValueError("id словаря не совпадает с именем файла")
        return pack

    def load(self, pack_id: str) -> DomainPackIn:
        path = self._path(pack_id)
        if not path.is_file():
            raise AppError(ErrorCode.DOMAIN_PACK_NOT_FOUND)
        try:
            return self._read(path)
        except (ValueError, ValidationError, OSError) as exc:
            raise AppError(ErrorCode.DOMAIN_PACK_INVALID, "Словарь повреждён: " + _short_error(exc)) from exc

    def compatibility(self, pack: DomainPackIn, model: LoadedModel | None) -> Compatibility:
        if model is None:
            return Compatibility(model_id=None, checked=False, compatible=None, missing_labels=[])
        missing = missing_labels(pack, model)
        return Compatibility(model_id=model.id, checked=True, compatible=not missing, missing_labels=missing)

    def to_out(self, pack: DomainPackIn, model: LoadedModel | None, active_id: str | None) -> DomainPackOut:
        return DomainPackOut(
            **pack.model_dump(), compatibility=self.compatibility(pack, model), is_active=pack.id == active_id
        )

    def list(
        self, model: LoadedModel | None, active_id: str | None
    ) -> list[DomainPackOut | DomainPackBroken]:
        result: list[DomainPackOut | DomainPackBroken] = []
        if not self.root.is_dir():
            return result
        for path in sorted(self.root.glob("*.json")):
            try:
                result.append(self.to_out(self._read(path), model, active_id))
            except (ValueError, ValidationError, OSError) as exc:
                result.append(DomainPackBroken(id=path.stem, error=_short_error(exc)))
        return result

    def _check_against_model(self, pack: DomainPackIn, model: LoadedModel | None) -> None:
        if model is None:
            return  # checked again when the pack is used with a model
        missing = missing_labels(pack, model)
        if missing:
            raise AppError(
                ErrorCode.DOMAIN_PACK_INVALID,
                f"Labels отсутствуют в активной модели {model.id}",
                details={"model_id": model.id, "missing_labels": missing},
            )

    def _write(self, pack: DomainPackIn) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = json.dumps(pack.model_dump(mode="json"), ensure_ascii=False, indent=2)
        fd, tmp = tempfile.mkstemp(dir=self.root, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(data + "\n")
            os.replace(tmp, self._path(pack.id))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def create(self, pack: DomainPackIn, model: LoadedModel | None) -> DomainPackIn:
        if self._path(pack.id).exists():
            raise AppError(ErrorCode.CONFLICT, f"Словарь {pack.id} уже существует")
        self._check_against_model(pack, model)
        self._write(pack)
        logger.info("Domain pack %s created", pack.id)
        return pack

    def update(self, pack_id: str, pack: DomainPackIn, model: LoadedModel | None) -> DomainPackIn:
        if pack.id != pack_id:
            raise AppError(ErrorCode.VALIDATION_ERROR, "id в теле запроса не совпадает с URL")
        if not self._path(pack_id).exists():
            raise AppError(ErrorCode.DOMAIN_PACK_NOT_FOUND)
        self._check_against_model(pack, model)
        self._write(pack)
        logger.info("Domain pack %s updated", pack.id)
        return pack

    def delete(self, pack_id: str, active_id: str | None) -> None:
        path = self._path(pack_id)
        if not path.exists():
            raise AppError(ErrorCode.DOMAIN_PACK_NOT_FOUND)
        if pack_id == active_id:
            raise AppError(ErrorCode.CONFLICT, "Нельзя удалить активный словарь — сначала отключите его")
        path.unlink()
        logger.info("Domain pack %s deleted", pack_id)

    def resolve(self, pack_id: str | None, model: LoadedModel) -> ResolvedPack:
        """Pack + class filter + phrase composer for a recognition session. Strict validation."""
        if not pack_id:
            return ResolvedPack(None, ClassFilter(model), PhraseComposer())
        pack = self.load(pack_id)
        self._check_against_model(pack, model)
        display = {model.label_index[normalize_label(label)]: label for label in pack.enabled_labels}
        class_filter = ClassFilter(model, pack.enabled_labels, display)
        if not class_filter.mask.any():
            raise AppError(
                ErrorCode.DOMAIN_PACK_INVALID, "Словарь не содержит ни одного распознаваемого класса"
            )
        templates = [PhraseTemplate(tuple(t.tokens), t.result) for t in pack.phrase_templates]
        return ResolvedPack(pack, class_filter, PhraseComposer(templates, pack.review_status))


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:3])
    if isinstance(exc, json.JSONDecodeError):
        return "некорректный JSON"
    return str(exc) or type(exc).__name__
