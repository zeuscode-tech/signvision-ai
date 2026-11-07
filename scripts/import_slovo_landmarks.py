#!/usr/bin/env python3
"""Import explicitly selected static Slovo classes into SignVision SQLite."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import ensure_directories, get_settings  # noqa: E402
from app.database import Database  # noqa: E402
from app.database.migrations import upgrade_to_head  # noqa: E402
from app.db_models import TrainingClass, TrainingSample, TrainingSession  # noqa: E402
from app.training.features import FeatureError, extract_static_features  # noqa: E402


def annotations(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096); handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        rows = {}
        for row in csv.DictReader(handle, dialect=dialect):
            key = (row.get("attachment_id") or "").strip()
            if key: rows[key] = row
        return rows


def hand_values(points) -> list[float] | None:
    if not isinstance(points, list) or len(points) != 21:
        return None
    values = []
    for point in points:
        values.extend([float(point["x"]), float(point["y"]), float(point["z"])])
    return values


def variants(frame: dict) -> list[list[float]]:
    hands = [hand_values(value) for key, value in sorted(frame.items()) if key.startswith("hand")]
    hands = [value for value in hands if value]
    zeros = [0.0] * 63
    if len(hands) == 1:
        return [hands[0] + zeros, zeros + hands[0]]
    if len(hands) >= 2:
        return [hands[0] + hands[1], hands[1] + hands[0]]
    return []


def json_object_items(path: Path):
    """Stream key/value pairs from one large top-level JSON object."""
    decoder=json.JSONDecoder();buffer="";position=0;started=False;finished=False
    with path.open("r",encoding="utf-8") as handle:
        while not finished:
            chunk=handle.read(1024*1024)
            if chunk: buffer=buffer[position:]+chunk;position=0
            elif position>=len(buffer): break
            while True:
                while position<len(buffer) and buffer[position].isspace():position+=1
                if not started:
                    if position>=len(buffer):break
                    if buffer[position]!="{":raise ValueError("Ожидался JSON-объект")
                    started=True;position+=1;continue
                while position<len(buffer) and (buffer[position].isspace() or buffer[position]==","):position+=1
                if position<len(buffer) and buffer[position]=="}":finished=True;break
                try:key,next_pos=decoder.raw_decode(buffer,position)
                except json.JSONDecodeError:break
                cursor=next_pos
                while cursor<len(buffer) and buffer[cursor].isspace():cursor+=1
                if cursor>=len(buffer):break
                if buffer[cursor] != ":":raise ValueError("Некорректный JSON")
                cursor+=1
                while cursor<len(buffer) and buffer[cursor].isspace():cursor+=1
                try:value,next_pos=decoder.raw_decode(buffer,cursor)
                except json.JSONDecodeError:break
                position=next_pos;yield str(key),value
            if not chunk and not finished:raise ValueError("Неожиданный конец JSON")


def main():
    parser=argparse.ArgumentParser(description="Импорт выбранных статических классов Slovo")
    parser.add_argument("--landmarks",type=Path,default=ROOT/"datasets/slovo/slovo_mediapipe.json")
    parser.add_argument("--annotations",type=Path,default=ROOT/"datasets/slovo/annotations.csv")
    parser.add_argument("--classes",help="Классы через запятую; выбирайте только действительно статические жесты")
    parser.add_argument("--list-classes",action="store_true")
    parser.add_argument("--frames-per-video",type=int,default=3,choices=range(1,9))
    parser.add_argument("--max-videos-per-class",type=int,default=20)
    args=parser.parse_args()
    meta=annotations(args.annotations)
    available=sorted({row.get("text","").strip() for row in meta.values() if row.get("text")})
    if args.list_classes:
        print("\n".join(available)); return
    if not args.classes: parser.error("--classes обязателен: динамические жесты нельзя автоматически превращать в статические")
    selected={value.strip() for value in args.classes.split(",") if value.strip()}
    missing=selected-set(available)
    if missing: parser.error("Классы не найдены: "+", ".join(sorted(missing)))
    settings=get_settings(); ensure_directories(settings)
    database=Database(settings.resolved_database_url); upgrade_to_head(database.engine)
    db=database.session_factory(); counts={name:0 for name in selected}; sessions={}; samples=0
    try:
        classes={}
        for name in selected:
            item=db.scalar(select(TrainingClass).where(TrainingClass.normalized_name==name.casefold()))
            if not item:
                item=TrainingClass(name=name,normalized_name=name.casefold(),
                                   description="Предварительный импорт Slovo landmarks; проверьте, что жест статический")
                db.add(item); db.flush()
            classes[name]=item
        for attachment_id, frames in json_object_items(args.landmarks):
            row = meta.get(attachment_id)
            name = (row or {}).get("text", "").strip()
            if name not in selected or counts[name] >= args.max_videos_per_class or not frames:
                continue
            signer = (row.get("user_id") or attachment_id).strip()
            key = (name, signer)
            if key not in sessions:
                session = TrainingSession(class_id=classes[name].id, note=f"Slovo: signer={signer}"[:200])
                db.add(session)
                db.flush()
                sessions[key] = session
            indexes = np.linspace(0, len(frames) - 1, args.frames_per_video, dtype=int)
            for index in indexes:
                for raw in variants(frames[int(index)]):
                    try:
                        feature = extract_static_features(raw)
                    except FeatureError:
                        continue
                    db.add(
                        TrainingSample(
                            session_id=sessions[key].id,
                            kind="static",
                            features_json=json.dumps(feature.tolist()),
                            frame_count=1,
                        )
                    )
                    samples += 1
            counts[name] += 1
            if samples and samples % 100 == 0:
                db.commit()
            if all(counts[class_name] >= args.max_videos_per_class for class_name in selected):
                break
        db.commit()
    finally:
        db.close(); database.dispose()
    print(json.dumps({"videos":counts,"samples":samples,"classes":sorted(selected)},ensure_ascii=False,indent=2))
    print("Теперь запустите приложение и нажмите «Обучить модель» в AI Studio.")


if __name__=="__main__": main()

