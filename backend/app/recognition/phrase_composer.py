"""Collect accepted gloss tokens and match manually reviewed phrase templates.

Recognised tokens are glosses, not a translation. A sentence is shown only
when it matches a template that exists in the domain pack; templates are never
generated automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model_loader import normalize_label

MAX_TOKENS = 40


@dataclass(frozen=True)
class PhraseTemplate:
    tokens: tuple[str, ...]
    result: str


class PhraseComposer:
    def __init__(
        self, templates: list[PhraseTemplate] | None = None, review_status: str = "unreviewed"
    ) -> None:
        self.templates = sorted(templates or [], key=lambda item: len(item.tokens), reverse=True)
        self.review_status = review_status
        self.tokens: list[str] = []

    def add(self, token: str) -> None:
        self.tokens.append(token)
        if len(self.tokens) > MAX_TOKENS:
            del self.tokens[: len(self.tokens) - MAX_TOKENS]

    def replace_last(self, token: str) -> None:
        if self.tokens:
            self.tokens[-1] = token
        else:
            self.tokens.append(token)

    def undo(self) -> None:
        if self.tokens:
            self.tokens.pop()

    def clear(self) -> None:
        self.tokens.clear()

    def match(self) -> PhraseTemplate | None:
        normalized = [normalize_label(token) for token in self.tokens]
        for template in self.templates:
            size = len(template.tokens)
            if size <= len(normalized) and normalized[-size:] == [
                normalize_label(t) for t in template.tokens
            ]:
                return template
        return None

    def snapshot(self) -> dict[str, Any]:
        template = self.match()
        return {
            "type": "phrase",
            "tokens": list(self.tokens),
            "gloss": " ".join(self.tokens),
            "template_text": template.result if template else None,
            "template_tokens": list(template.tokens) if template else None,
            "template_review_status": self.review_status if template else None,
            "is_translation": False,
        }
