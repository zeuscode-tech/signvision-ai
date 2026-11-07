from __future__ import annotations

from app.recognition.gesture_segmenter import GestureSegmenter
from app.recognition.stabilization import Stabilizer, StabilizerConfig


def stabilizer(
    threshold: float = 0.65, stability: int = 2, cooldown: int = 1000, pause: int = 500
) -> Stabilizer:
    return Stabilizer(StabilizerConfig(threshold, stability, cooldown, pause))


def test_low_confidence_is_uncertain_and_never_accepted() -> None:
    s = stabilizer()
    for t in range(5):
        decision = s.observe("A", 0.5, t * 100)
        assert decision.state == "uncertain" and not decision.accepted


def test_needs_n_consecutive_inferences() -> None:
    s = stabilizer(stability=3)
    assert s.observe("A", 0.9, 0).reason == "awaiting_stability"
    assert s.observe("A", 0.9, 100).reason == "awaiting_stability"
    assert s.observe("A", 0.9, 200).accepted


def test_class_change_resets_streak() -> None:
    s = stabilizer()
    s.observe("A", 0.9, 0)
    assert not s.observe("B", 0.9, 100).accepted
    assert s.observe("B", 0.9, 200).accepted


def test_held_gesture_is_accepted_once() -> None:
    s = stabilizer(cooldown=0, pause=10_000)
    accepted = [s.observe("A", 0.95, t * 100).accepted for t in range(30)]
    assert accepted.count(True) == 1
    assert s.observe("A", 0.95, 3100).reason == "duplicate_suppressed"


def test_cooldown_blocks_next_word() -> None:
    s = stabilizer(cooldown=1000)
    s.observe("A", 0.9, 0)
    assert s.observe("A", 0.9, 100).accepted
    s.observe("B", 0.9, 200)
    assert s.observe("B", 0.9, 300).state == "cooldown"
    assert s.observe("B", 0.9, 1200).accepted


def test_same_word_again_after_release() -> None:
    s = stabilizer(cooldown=0)
    s.observe("A", 0.9, 0)
    assert s.observe("A", 0.9, 100).accepted
    s.release()  # hands disappeared / model said "no gesture"
    s.observe("A", 0.9, 200)
    assert s.observe("A", 0.9, 300).accepted


def test_same_word_again_after_pause() -> None:
    s = stabilizer(cooldown=0, pause=500)
    s.observe("A", 0.9, 0)
    assert s.observe("A", 0.9, 100).accepted
    s.observe(None, 0.0, 300)
    s.observe("A", 0.9, 700)
    assert s.observe("A", 0.9, 800).accepted


def test_segmenter_detects_pause_once() -> None:
    seg = GestureSegmenter(pause_ms=300)
    assert seg.idle
    assert seg.update(True, 0) == "hands"
    assert seg.update(False, 100) == "hands_missing"
    assert seg.update(False, 450) == "pause_started"
    assert seg.update(False, 500) == "idle"
    assert seg.update(None, 600) == "unknown" and not seg.idle
