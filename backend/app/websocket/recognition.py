"""WebSocket /ws/recognition: binary JPEG frames in, honest predictions out.

Protocol
  client → {"type": "start", ...}                 server → {"type": "ready", ...}
  client → binary JPEG frames                       server → {"type": "prediction", "state": ...}
  client → {"type": "correct", "event_id", "label"} / "phrase_undo" / "phrase_clear" / "reset" / "ping"
  client → {"type": "stop"}                         server → {"type": "stopped", "summary": ...}

Each connection owns its frame queue (drop-oldest), ring buffer, stabiliser,
hand detector and metrics. Decoding and inference run in thread pools so the
event loop stays responsive. On stop or disconnect the session is finished,
buffers are cleared and only aggregated technical metrics are persisted.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from typing import Any, TypeVar

import anyio
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from ..core.exceptions import AppError, ErrorCode
from ..core.logging import request_id_var
from ..core.security import ensure_within, is_loopback_client, is_origin_allowed
from ..core.timeutils import isoformat_z, utcnow
from ..dependencies import Container, get_container
from ..recognition.frame_buffer import DropOldestQueue, FrameBuffer
from ..recognition.gesture_segmenter import GestureSegmenter
from ..recognition.hand_detection import HandDetector, NullHandDetector
from ..recognition.inference import InferenceOutput
from ..recognition.model_loader import LoadedModel, normalize_label
from ..recognition.preprocessing import FramePreprocessor, FrameTooLargeError, InvalidFrameError
from ..recognition.stabilization import Stabilizer, StabilizerConfig
from ..repositories import settings_repo
from ..schemas.websocket import ControlMessage, StartMessage
from ..services import session_service
from ..services.domain_pack_service import ResolvedPack
from ..services.session_service import SessionMetrics

logger = logging.getLogger(__name__)
router = APIRouter()

START_TIMEOUT_S = 10
MAX_TEXT_BYTES = 4_096
QUEUE_SIZE = 2
NOTICE_INTERVAL_MS = 1_000
COLLECTING_EVERY = 4


T = TypeVar("T")


def first_set(*values: T | None) -> T:
    return next(value for value in values if value is not None)


def now_ms() -> int:
    return int(time.monotonic() * 1000)


class TokenBucket:
    def __init__(self, rate: float, capacity: float) -> None:
        self.rate, self.capacity = rate, capacity
        self.tokens = capacity
        self.updated = time.monotonic()

    def allow(self) -> bool:
        current = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (current - self.updated) * self.rate)
        self.updated = current
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


def prediction(state: str, **fields: Any) -> dict[str, Any]:
    message: dict[str, Any] = {
        "type": "prediction",
        "state": state,
        "label": None,
        "confidence": None,
        "top3": [],
        "accepted": False,
    }
    message.update(fields)
    message["timestamp"] = isoformat_z(utcnow())
    return message


class RecognitionConnection:
    def __init__(self, websocket: WebSocket, container: Container) -> None:
        self.ws = websocket
        self.container = container
        self.settings = container.settings
        self.connection_id = str(uuid.uuid4())
        self.metrics = SessionMetrics()
        self.session_id: str | None = None
        self.end_reason = "disconnect"
        self.model: LoadedModel | None = None
        self.pack: ResolvedPack | None = None
        self.detector: HandDetector = NullHandDetector()
        self.preprocessor: FramePreprocessor | None = None
        self.buffer: FrameBuffer | None = None
        self.stabilizer: Stabilizer | None = None
        self.segmenter = GestureSegmenter(self.settings.pause_ms)
        self.queue = DropOldestQueue(QUEUE_SIZE)
        self.frame_ready = asyncio.Event()
        self.rate = TokenBucket(self.settings.max_websocket_fps, max(1.0, self.settings.max_websocket_fps))
        self.send_lock = asyncio.Lock()
        self.processor: asyncio.Task[None] | None = None
        self.inference_task: asyncio.Task[None] | None = None
        self.last_state: str | None = None
        self.notices: dict[str, int] = {}
        self.closed = False

    # ---- transport helpers -------------------------------------------------------
    async def send(self, message: dict[str, Any]) -> None:
        if self.closed or self.ws.application_state != WebSocketState.CONNECTED:
            return
        async with self.send_lock:
            try:
                await self.ws.send_text(json.dumps(message, ensure_ascii=False))
            except (RuntimeError, WebSocketDisconnect):
                self.closed = True

    async def send_state(self, state: str, **fields: Any) -> None:
        self.last_state = state
        await self.send(prediction(state, **fields))

    async def notify(self, code: ErrorCode, message: str, *, throttle: bool = True) -> None:
        current = now_ms()
        if throttle and current - self.notices.get(code, -NOTICE_INTERVAL_MS) < NOTICE_INTERVAL_MS:
            return
        self.notices[code] = current
        await self.send(
            {
                "type": "error",
                "state": "error",
                "code": str(code),
                "message": message,
                "timestamp": isoformat_z(utcnow()),
            }
        )

    async def close(self, code: int = 1000) -> None:
        if not self.closed and self.ws.application_state == WebSocketState.CONNECTED:
            with contextlib.suppress(RuntimeError):
                await self.ws.close(code)
        self.closed = True

    async def shutdown(self) -> None:
        self.end_reason = "server_shutdown"
        await self.close(1001)

    # ---- lifecycle ------------------------------------------------------------------
    async def run(self) -> None:
        try:
            if not await self.handshake():
                return
            self.processor = asyncio.create_task(self.process_loop())
            self.processor.add_done_callback(_log_task_failure)
            await self.receive_loop()
        except WebSocketDisconnect:
            self.end_reason = "disconnect"
        except Exception:
            logger.exception("WebSocket connection failed")
            self.end_reason = "error"
            await self.notify(ErrorCode.INTERNAL_ERROR, "Внутренняя ошибка распознавания", throttle=False)
        finally:
            # Shielded: a cancelled connection task (server shutdown, client gone) must still
            # finish the session, release buffers and close the hand detector.
            with anyio.CancelScope(shield=True):
                await self.cleanup()

    async def handshake(self) -> bool:
        try:
            raw = await asyncio.wait_for(self.ws.receive_text(), timeout=START_TIMEOUT_S)
            start = StartMessage.model_validate_json(raw)
        except TimeoutError:
            await self.notify(ErrorCode.VALIDATION_ERROR, "Не получено сообщение start", throttle=False)
            await self.close(1008)
            return False
        except (ValidationError, ValueError, KeyError):
            await self.notify(
                ErrorCode.VALIDATION_ERROR, 'Первое сообщение должно быть {"type": "start"}', throttle=False
            )
            await self.close(1008)
            return False
        try:
            await self.setup(start)
        except AppError as exc:
            if exc.code in {ErrorCode.MODEL_NOT_FOUND, ErrorCode.MODEL_NOT_READY, ErrorCode.MODEL_INVALID}:
                await self.send_state(
                    "model_unavailable",
                    code=str(exc.code),
                    message=exc.message,
                    model_id=start.model_id or self.container.models.active_id,
                )
                await self.close(1013)
            else:
                await self.notify(exc.code, exc.message, throttle=False)
                await self.close(1008)
            return False
        assert self.model is not None
        await self.send(
            {
                "type": "ready",
                "session_id": self.session_id,
                "model_id": self.model.id,
                "required_frames": self.model.manifest.frame_count,
                "max_fps": self.settings.max_websocket_fps,
                "max_frame_bytes": self.settings.max_frame_size_bytes,
                "domain_pack": self.pack.pack.id if self.pack and self.pack.pack else None,
                "hand_detection": self.detector.available,
                "thresholds": {
                    "confidence": self.stabilizer.config.threshold if self.stabilizer else None,
                    "stability_count": self.stabilizer.config.stability_count if self.stabilizer else None,
                    "cooldown_ms": self.stabilizer.config.cooldown_ms if self.stabilizer else None,
                },
            }
        )
        await self.send_state("waiting", buffered=0, required=self.model.manifest.frame_count)
        return True

    async def setup(self, start: StartMessage) -> None:
        models = self.container.models
        model_id = start.model_id or models.active_id
        if model_id is None:
            raise AppError(ErrorCode.MODEL_NOT_READY)
        if model_id == models.active_id and models.active() is None:
            raise AppError(
                ErrorCode.MODEL_NOT_READY, f"Модель {model_id} недоступна: {models.active_status()}"
            )
        model = models.active() if model_id == models.active_id else None
        if model is None:
            model = await asyncio.to_thread(models.get, model_id)
        self.model = model

        def load_prefs_and_session() -> tuple[str | None, StabilizerConfig, str]:
            with self.container.database.session() as db:
                prefs = settings_repo.get_or_create(db)
                db.commit()
                pack_id = (
                    start.domain_pack
                    if "domain_pack" in start.model_fields_set
                    else prefs.active_domain_pack_id
                )
                manifest = model.manifest
                # Priority: user settings > model manifest > environment defaults.
                config = StabilizerConfig(
                    threshold=first_set(
                        prefs.confidence_threshold,
                        manifest.confidence_threshold,
                        self.settings.default_confidence_threshold,
                    ),
                    stability_count=first_set(
                        prefs.stability_count, manifest.stability_count, self.settings.default_stability_count
                    ),
                    cooldown_ms=first_set(
                        prefs.cooldown_ms, manifest.cooldown_ms, self.settings.default_cooldown_ms
                    ),
                    pause_ms=self.settings.pause_ms,
                )
                if start.session_id:
                    session = session_service.get(db, start.session_id)
                    if session.status != "active":
                        raise AppError(ErrorCode.SESSION_NOT_FOUND, "Сессия уже завершена")
                    session.model_id, session.domain_pack_id = model.id, pack_id
                    db.commit()
                    session_id = session.id
                else:
                    session_id = session_service.create(
                        db, mode=start.mode, model_id=model.id, domain_pack_id=pack_id
                    ).id
                return pack_id, config, session_id

        pack_id, config, session_id = await asyncio.to_thread(load_prefs_and_session)
        self.session_id = session_id  # set first so cleanup() finishes the session if setup fails below
        self.pack = self.container.packs.resolve(pack_id, model)  # strict: missing labels → error
        self.stabilizer = Stabilizer(config)
        self.buffer = FrameBuffer(model.manifest.frame_count)
        self.detector = await asyncio.to_thread(self.container.hands.create)
        self.preprocessor = FramePreprocessor(
            model.manifest.preprocessing, self.detector, self.settings.max_frame_size_bytes
        )
        logger.info(
            "Recognition session %s started (model=%s, pack=%s, hands=%s)",
            session_id,
            model.id,
            pack_id,
            self.detector.available,
        )

    async def receive_loop(self) -> None:
        while True:
            message = await self.ws.receive()
            if message["type"] == "websocket.disconnect":
                self.end_reason = "disconnect"
                return
            data = message.get("bytes")
            if data is not None:
                await self.on_frame(data)
                continue
            text = message.get("text")
            if text is not None and await self.on_control(text):
                return

    async def on_frame(self, data: bytes) -> None:
        self.metrics.frames_received += 1
        if len(data) > self.settings.max_frame_size_bytes:
            self.metrics.invalid_frames += 1
            await self.notify(
                ErrorCode.FRAME_TOO_LARGE,
                f"Кадр превышает {self.settings.max_frame_size_bytes} байт и отброшен",
            )
            return
        if not self.rate.allow():
            self.metrics.frames_dropped += 1
            await self.notify(
                ErrorCode.RATE_LIMIT_EXCEEDED,
                f"Частота кадров выше {self.settings.max_websocket_fps:g} FPS; лишние кадры отброшены",
            )
            return
        self.queue.put((data, now_ms()))
        self.frame_ready.set()

    async def on_control(self, text: str) -> bool:
        if len(text.encode()) > MAX_TEXT_BYTES:
            await self.notify(ErrorCode.VALIDATION_ERROR, "Слишком большое управляющее сообщение")
            return False
        try:
            control = ControlMessage.model_validate_json(text)
        except (ValidationError, ValueError):
            await self.notify(ErrorCode.VALIDATION_ERROR, "Неизвестное управляющее сообщение")
            return False
        if control.type == "stop":
            self.end_reason = "stop"
            return True
        if control.type == "ping":
            await self.send({"type": "pong", "timestamp": isoformat_z(utcnow())})
        elif control.type == "reset":
            self.reset_window()
            await self.send_state(
                "waiting", buffered=0, required=self.buffer.capacity if self.buffer is not None else 0
            )
        elif control.type in {"phrase_undo", "phrase_clear"} and self.pack is not None:
            if control.type == "phrase_undo":
                self.pack.composer.undo()
            else:
                self.pack.composer.clear()
            await self.send(self.pack.composer.snapshot())
        elif control.type == "correct":
            await self.on_correction(control)
        return False

    async def on_correction(self, control: ControlMessage) -> None:
        assert self.model is not None and self.pack is not None and self.session_id is not None
        label = (control.label or "").strip()
        index = self.model.label_index.get(normalize_label(label))
        if control.event_id is None or index is None or not self.pack.class_filter.mask[index]:
            await self.notify(
                ErrorCode.VALIDATION_ERROR,
                "Исправление должно указывать event_id и допустимый label",
                throttle=False,
            )
            return
        display = self.pack.class_filter.label(index)
        try:
            event = await asyncio.to_thread(self._correct_event, control.event_id, display)
        except AppError as exc:
            await self.notify(exc.code, exc.message, throttle=False)
            return
        self.metrics.corrections += 1
        if event["accepted"]:
            self.pack.composer.replace_last(display)
        else:
            self.pack.composer.add(display)
        await self.send({"type": "corrected", "event_id": control.event_id, "label": display})
        await self.send(self.pack.composer.snapshot())

    def _correct_event(self, event_id: int, label: str) -> dict[str, Any]:
        assert self.session_id is not None
        with self.container.database.session() as db:
            event = session_service.correct_event(db, self.session_id, event_id, label)
            return {"accepted": event.accepted}

    def reset_window(self) -> None:
        if self.buffer is not None:
            self.buffer.clear()
        if self.stabilizer is not None:
            self.stabilizer.release()

    # ---- frame processing -------------------------------------------------------
    async def process_loop(self) -> None:
        while True:
            await self.frame_ready.wait()
            item = self.queue.get()
            if item is None:
                self.frame_ready.clear()
                continue
            try:
                await self.process_frame(*item)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Frame processing failed")
                await self.notify(ErrorCode.INTERNAL_ERROR, "Ошибка обработки кадра")

    async def process_frame(self, data: bytes, received_ms: int) -> None:
        assert self.preprocessor is not None and self.buffer is not None and self.model is not None
        try:
            frame = await self.container.preprocess.run(self.preprocessor.process, data, received_ms)
        except FrameTooLargeError:
            self.metrics.invalid_frames += 1
            await self.notify(ErrorCode.FRAME_TOO_LARGE, "Кадр превышает допустимый размер")
            return
        except InvalidFrameError as exc:
            self.metrics.invalid_frames += 1
            await self.notify(ErrorCode.INVALID_FRAME, str(exc))
            return
        self.metrics.frames_processed += 1
        if self.settings.save_raw_frames:
            await asyncio.to_thread(self._save_raw_frame, data)
        # The window slides over every frame, so when a gesture starts it already holds the
        # preceding context and a short sign is complete in the window right after it ends.
        self.buffer.append(frame.image)
        event = self.segmenter.update(frame.hands.present, received_ms)
        if event == "pause_started":
            # Hands gone for PAUSE_MS: the gesture is over; the same word may be accepted again.
            if self.stabilizer is not None:
                self.stabilizer.release()
            await self.send_state("no_hands")
            return
        if event == "idle":  # no hands: skip inference to save CPU
            if self.last_state != "no_hands":
                await self.send_state("no_hands")
            return
        if not self.buffer.is_full:
            if self.last_state != "collecting" or len(self.buffer) % COLLECTING_EVERY == 0:
                await self.send_state("collecting", buffered=len(self.buffer), required=self.buffer.capacity)
            return
        inference_idle = self.inference_task is None or self.inference_task.done()
        if inference_idle and self.buffer.ready_for_inference(self.model.manifest.stride):
            frames = self.buffer.snapshot()
            await self.send_state("processing", buffered=len(frames), required=self.buffer.capacity)
            self.inference_task = asyncio.create_task(self.run_inference(frames))
            self.inference_task.add_done_callback(_log_task_failure)

    def _save_raw_frame(self, data: bytes) -> None:
        # Only when SAVE_RAW_FRAMES=true (off by default).
        directory = ensure_within(self.settings.data_dir, self.settings.raw_frame_dir / str(self.session_id))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{self.metrics.frames_processed:06d}.jpg").write_bytes(data)

    async def run_inference(self, frames: list[np.ndarray]) -> None:
        assert self.model is not None and self.pack is not None and self.stabilizer is not None
        try:
            output = await self.container.inference.infer(
                self.model, frames, self.model.manifest.preprocessing
            )
        except Exception:
            logger.exception("Inference failed")
            await self.notify(ErrorCode.INTERNAL_ERROR, "Ошибка инференса модели")
            return
        finally:
            frames.clear()
        self.metrics.predictions += 1
        self.metrics.add_latency(output.inference_ms)
        message = self.evaluate(output)
        if message["state"] in {"recognized", "uncertain", "cooldown"}:
            message["event_id"] = await asyncio.to_thread(
                self._record_event,
                message["state"],
                message.get("candidate"),
                float(message["confidence"]),
                output.inference_ms,
                bool(message["accepted"]),
            )
        message.pop("candidate", None)
        self.last_state = message["state"]
        await self.send(message)
        if message["accepted"]:
            self.metrics.accepted += 1
            self.pack.composer.add(message["label"])
            await self.send(self.pack.composer.snapshot())

    def evaluate(self, output: InferenceOutput) -> dict[str, Any]:
        assert self.model is not None and self.pack is not None and self.stabilizer is not None
        probabilities = output.probabilities
        inference_ms = round(output.inference_ms, 1)
        overall = int(np.argmax(probabilities))
        if self.model.no_event_index is not None and overall == self.model.no_event_index:
            self.stabilizer.release()
            return prediction(
                "waiting",
                confidence=round(float(probabilities[overall]), 4),
                reason="no_event",
                inference_ms=inference_ms,
            )
        candidates = self.pack.class_filter.top_k(probabilities, 3)
        best = candidates[0] if candidates else None
        decision = self.stabilizer.observe(
            best.label if best else None, best.confidence if best else 0.0, now_ms()
        )
        confidence = round(best.confidence, 4) if best else 0.0
        if decision.state == "uncertain" or best is None:
            # Never force a class when the model is not confident enough.
            return prediction(
                "uncertain",
                confidence=confidence,
                reason=decision.reason,
                stable_count=0,
                inference_ms=inference_ms,
                candidate=best.label if best else None,
            )
        return prediction(
            decision.state,
            label=best.label,
            confidence=confidence,
            accepted=decision.accepted,
            top3=[{"label": c.label, "confidence": round(c.confidence, 4)} for c in candidates],
            stable_count=decision.stable_count,
            reason=decision.reason,
            inference_ms=inference_ms,
            candidate=best.label,
        )

    def _record_event(
        self, state: str, label: str | None, confidence: float, inference_ms: float, accepted: bool
    ) -> int | None:
        assert self.session_id is not None
        with self.container.database.session() as db:
            return session_service.record_event(
                db,
                self.session_id,
                state=state,
                predicted_label=label,
                confidence=confidence,
                inference_ms=inference_ms,
                accepted=accepted,
            )

    # ---- teardown ---------------------------------------------------------------
    async def cleanup(self) -> None:
        if self.processor is not None:
            self.processor.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self.processor
        if self.inference_task is not None and not self.inference_task.done():
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(self.inference_task), timeout=5)
        self.metrics.frames_dropped += self.queue.dropped
        self.queue.clear()
        if self.buffer is not None:
            self.buffer.clear()
        if self.pack is not None:
            self.pack.composer.clear()
        await asyncio.to_thread(self.detector.close)
        summary: dict[str, Any] | None = None
        if self.session_id is not None:
            summary = await asyncio.to_thread(self._finish_session)
        if self.end_reason == "stop":
            await self.send({"type": "stopped", "session_id": self.session_id, "summary": summary})
            await self.close(1000)
        await self.container.connections.unregister(self)

    def _finish_session(self) -> dict[str, Any] | None:
        assert self.session_id is not None
        with self.container.database.session() as db:
            try:
                session = session_service.finish(
                    db, self.session_id, reason=self.end_reason, metrics=self.metrics
                )
            except AppError:
                return None  # session row removed (e.g. full data deletion)
            logger.info(
                "Recognition session %s finished (%s): %d frames, %d predictions, p95 %s ms",
                session.id,
                self.end_reason,
                session.frames_processed,
                session.predictions_count,
                session.p95_latency_ms,
            )
            return {
                "duration_ms": session.duration_ms,
                "frames_received": session.frames_received,
                "frames_processed": session.frames_processed,
                "frames_dropped": session.frames_dropped,
                "invalid_frames": session.invalid_frames,
                "predictions": session.predictions_count,
                "accepted": session.accepted_count,
                "corrections": session.corrections_count,
                "avg_latency_ms": session.avg_latency_ms,
                "median_latency_ms": session.median_latency_ms,
                "p95_latency_ms": session.p95_latency_ms,
            }


def _log_task_failure(task: asyncio.Task[None]) -> None:
    if not task.cancelled() and task.exception() is not None:
        logger.error("Recognition task failed", exc_info=task.exception())


@router.websocket("/ws/recognition")
async def recognition_socket(websocket: WebSocket) -> None:
    container = get_container(websocket)
    request_id_var.set(str(uuid.uuid4()))
    client_host = websocket.client.host if websocket.client else None
    origin = websocket.headers.get("origin")
    if not is_loopback_client(client_host) or not is_origin_allowed(origin, container.settings):
        logger.warning("Rejected WebSocket connection (origin not allowed)")
        await websocket.close(code=1008)
        return
    await websocket.accept()
    connection = RecognitionConnection(websocket, container)
    if not await container.connections.register(connection):
        await connection.notify(
            ErrorCode.RATE_LIMIT_EXCEEDED, "Слишком много одновременных подключений", throttle=False
        )
        await connection.close(1013)
        return
    await connection.run()
