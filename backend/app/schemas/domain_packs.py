from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from .common import ApiModel
from .manifest import MODEL_ID_PATTERN


def _norm(value: str) -> str:
    return value.strip().casefold()


class PhraseTemplateIn(ApiModel):
    tokens: list[str] = Field(min_length=1, max_length=10)
    result: str = Field(min_length=1, max_length=500)


class DomainPackIn(ApiModel):
    id: str = Field(pattern=MODEL_ID_PATTERN)
    name: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=40)
    description: str = Field(default="", max_length=2000)
    enabled_labels: list[str] = Field(min_length=1, max_length=2000)
    phrase_templates: list[PhraseTemplateIn] = Field(default_factory=list, max_length=500)
    quick_responses: list[str] = Field(default_factory=list, max_length=200)
    # Templates and gloss mappings must be checked by a qualified RSL reviewer.
    review_status: Literal["unreviewed", "reviewed"] = "unreviewed"
    reviewed_by: str | None = Field(default=None, max_length=200)

    @field_validator("enabled_labels")
    @classmethod
    def labels_unique(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("Пустой label недопустим")
        if len({_norm(item) for item in cleaned}) != len(cleaned):
            raise ValueError("enabled_labels содержит повторы")
        return cleaned

    @field_validator("quick_responses")
    @classmethod
    def responses_not_empty(cls, value: list[str]) -> list[str]:
        if any(not item.strip() or len(item) > 500 for item in value):
            raise ValueError("Быстрый ответ должен быть непустой строкой до 500 символов")
        return value

    @model_validator(mode="after")
    def templates_use_enabled_labels(self) -> DomainPackIn:
        enabled = {_norm(item) for item in self.enabled_labels}
        for template in self.phrase_templates:
            unknown = [token for token in template.tokens if _norm(token) not in enabled]
            if unknown:
                raise ValueError(f"Шаблон использует labels вне enabled_labels: {', '.join(unknown)}")
        return self


class Compatibility(ApiModel):
    model_id: str | None
    checked: bool
    compatible: bool | None
    missing_labels: list[str]


class DomainPackOut(DomainPackIn):
    status: Literal["valid", "invalid"] = "valid"
    compatibility: Compatibility | None = None
    is_active: bool = False


class DomainPackBroken(ApiModel):
    id: str
    status: Literal["invalid"] = "invalid"
    error: str
