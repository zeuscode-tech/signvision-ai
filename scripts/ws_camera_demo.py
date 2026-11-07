#!/usr/bin/env python3
"""Manual check of /ws/recognition with a real webcam (no frontend needed).

    .venv/bin/python scripts/ws_camera_demo.py --domain-pack medical

macOS asks for camera permission for the terminal on first run. Frames go only
to the local backend; press Ctrl+C to stop and print the session summary.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import cv2
import websockets


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws/recognition")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--domain-pack", default=None)
    parser.add_argument("--fps", type=float, default=12)
    args = parser.parse_args()
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit("Камера недоступна: проверьте разрешение в System Settings → Privacy & Security → Camera")
    async with websockets.connect(args.url) as ws:
        start = {"type": "start", "domain_pack": args.domain_pack} if args.domain_pack else {"type": "start"}
        await ws.send(json.dumps(start))
        print(await ws.recv())

        async def reader() -> None:
            async for raw in ws:
                message = json.loads(raw)
                if message.get("accepted"):
                    print(f"✔ {message['label']}  {message['confidence']:.0%}  top3={message['top3']}")
                elif message.get("type") == "phrase":
                    print("  фраза (глоссы):", message["gloss"], "| шаблон:", message["template_text"])
                elif message.get("state") in {"uncertain", "no_hands", "model_unavailable"} or message["type"] in {
                        "error", "stopped"}:
                    print(" ", message.get("state") or message["type"], message.get("message", ""),
                          message.get("summary", ""))
                if message.get("type") == "stopped":
                    return

        task = asyncio.create_task(reader())
        try:
            while True:
                ok, frame = capture.read()
                if ok:
                    frame = cv2.resize(frame, (640, 480))
                    await ws.send(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
                await asyncio.sleep(1 / args.fps)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            capture.release()
            await ws.send(json.dumps({"type": "stop"}))
            await asyncio.wait_for(task, timeout=10)
            time.sleep(0.1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
