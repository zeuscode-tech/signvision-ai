import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "import_slovo_landmarks.py"
SPEC = importlib.util.spec_from_file_location("import_slovo_landmarks", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
json_object_items = MODULE.json_object_items
variants = MODULE.variants


def points(offset: float = 0.0) -> list[dict[str, float]]:
    return [{"x": offset + index / 100, "y": index / 200, "z": -index / 300} for index in range(21)]


def test_streams_top_level_landmark_object(tmp_path):
    source = tmp_path / "landmarks.json"
    source.write_text(json.dumps({"clip-a": [{"hand 1": points()}], "clip-b": []}), encoding="utf-8")

    assert list(json_object_items(source)) == [
        ("clip-a", [{"hand 1": points()}]),
        ("clip-b", []),
    ]


def test_one_hand_is_imported_for_both_handed_slots():
    rows = variants({"hand 1": points()})

    assert len(rows) == 2
    assert len(rows[0]) == 126
    assert rows[0][63:] == [0.0] * 63
    assert rows[1][:63] == [0.0] * 63
