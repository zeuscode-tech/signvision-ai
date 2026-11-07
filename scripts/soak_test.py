#!/usr/bin/env python3
"""Long-running local load test for /ws/recognition.

Streams JPEG frames at a fixed FPS for N minutes, reconnects after every
`--session-seconds`, and samples the backend's RSS memory. Prints a JSON
report with latency (mean / median / p95), reconnect time and memory growth.

    .venv/bin/python scripts/soak_test.py --minutes 30 --pid $(pgrep -f "uvicorn app.main")
    .venv/bin/python scripts/soak_test.py --minutes 30 --video sample.mp4   # real footage
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import websockets


def rss_mb(pid: int | None) -> float | None:
    if pid is None:
        return None
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=False)
    return round(int(out.stdout.strip()) / 1024, 1) if out.stdout.strip() else None


def frames(video: Path | None) -> list[bytes]:
    if video is None:
        rng = np.random.default_rng(0)
        images = [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(30)]
    else:
        capture, images = cv2.VideoCapture(str(video)), []
        while len(images) < 600:
            ok, image = capture.read()
            if not ok:
                break
            images.append(cv2.resize(image, (640, 480)))
    return [cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes() for image in images]


async def session(url: str, payload: list[bytes], fps: float, seconds: float, stats: dict) -> None:  # type: ignore[type-arg]
    started = time.perf_counter()
    async with websockets.connect(url, max_size=2**20) as ws:
        await ws.send(json.dumps({"type": "start"}))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            raise RuntimeError(f"backend not ready: {ready}")
        stats["connect_ms"].append((time.perf_counter() - started) * 1000)

        async def reader() -> None:
            async for raw in ws:
                message = json.loads(raw)
                if "inference_ms" in message:
                    stats["inference_ms"].append(message["inference_ms"])
                    stats["states"][message["state"]] = stats["states"].get(message["state"], 0) + 1
                if message.get("type") == "stopped":
                    stats["sessions"].append(message["summary"])
                    return

        task = asyncio.create_task(reader())
        end, index = time.perf_counter() + seconds, 0
        while time.perf_counter() < end:
            await ws.send(payload[index % len(payload)])
            index += 1
            await asyncio.sleep(1 / fps)
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(task, timeout=10)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/recognition")
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--fps", type=float, default=12)
    parser.add_argument("--session-seconds", type=float, default=120)
    parser.add_argument("--pid", type=int, default=None, help="backend PID for memory sampling")
    parser.add_argument("--video", type=Path, default=None)
    args = parser.parse_args()
    payload = frames(args.video)
    stats: dict = {"inference_ms": [], "connect_ms": [], "states": {}, "sessions": [], "rss_mb": []}  # type: ignore[type-arg]
    deadline = time.perf_counter() + args.minutes * 60
    stats["rss_mb"].append(rss_mb(args.pid))
    while time.perf_counter() < deadline:
        await session(args.url, payload, args.fps, min(args.session_seconds, deadline - time.perf_counter()), stats)
        stats["rss_mb"].append(rss_mb(args.pid))
    latencies = sorted(stats["inference_ms"])
    report = {
        "minutes": args.minutes, "fps": args.fps, "sessions": len(stats["sessions"]),
        "predictions": len(latencies), "states": stats["states"],
        "inference_ms": {"mean": round(statistics.fmean(latencies), 1), "median": round(statistics.median(latencies), 1),
                         "p95": round(latencies[int(len(latencies) * 0.95) - 1], 1)} if latencies else None,
        "reconnect_ms": {"max": round(max(stats["connect_ms"]), 1), "median": round(statistics.median(stats["connect_ms"]), 1)},
        "frames_dropped": sum(s["frames_dropped"] for s in stats["sessions"]),
        "rss_mb": stats["rss_mb"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
