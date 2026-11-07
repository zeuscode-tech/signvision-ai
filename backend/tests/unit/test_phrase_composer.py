from __future__ import annotations

from app.recognition.phrase_composer import PhraseComposer, PhraseTemplate


def test_matches_reviewed_template_on_tail_case_insensitively() -> None:
    composer = PhraseComposer([PhraseTemplate(("БОЛЬ", "ГОЛОВА"), "У меня болит голова")], "reviewed")
    for token in ("врач", "боль", "голова"):
        composer.add(token)
    snapshot = composer.snapshot()
    assert snapshot["template_text"] == "У меня болит голова"
    assert snapshot["template_review_status"] == "reviewed"
    assert snapshot["gloss"] == "врач боль голова"
    assert snapshot["is_translation"] is False


def test_no_template_means_no_sentence() -> None:
    composer = PhraseComposer([PhraseTemplate(("A", "B"), "text")])
    composer.add("B")
    composer.add("A")
    assert composer.snapshot()["template_text"] is None


def test_longest_template_wins_and_undo() -> None:
    composer = PhraseComposer([PhraseTemplate(("B",), "short"), PhraseTemplate(("A", "B"), "long")])
    composer.add("A")
    composer.add("B")
    assert composer.match() and composer.match().result == "long"  # type: ignore[union-attr]
    composer.undo()
    assert composer.tokens == ["A"]
    composer.replace_last("B")
    assert composer.tokens == ["B"]
