"""Hand presence and hand boxes for crop, `no_hands` and gesture pauses.

Server-side detection uses the same MediaPipe Hand Landmarker file the
frontend already ships. If MediaPipe or the file is missing, `NullHandDetector`
reports "unknown" and the pipeline relies on the model's own no-event class.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DETECTION_MAX_SIDE = 320


@dataclass
class HandObservation:
    # True / False, or None when no detector is available.
    present: bool | None
    # Normalised (x0, y0, x1, y1) boxes in 0..1 image coordinates.
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)


class HandDetector(Protocol):
    available: bool

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> HandObservation: ...

    def close(self) -> None: ...


class NullHandDetector:
    available = False

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> HandObservation:
        return HandObservation(present=None)

    def close(self) -> None:
        return None


class MediaPipeHandDetector:
    """One instance per WebSocket connection (VIDEO mode keeps tracking state)."""

    available = True

    def __init__(self, model_path: Path) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1
        self._lock = threading.Lock()

    def detect(self, rgb: np.ndarray, timestamp_ms: int) -> HandObservation:
        height, width = rgb.shape[:2]
        scale = DETECTION_MAX_SIDE / max(height, width)
        small = cv2.resize(rgb, (round(width * scale), round(height * scale))) if scale < 1 else rgb
        with self._lock:
            # VIDEO mode requires strictly increasing timestamps.
            timestamp_ms = max(timestamp_ms, self._last_ts + 1)
            self._last_ts = timestamp_ms
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(small))
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        boxes: list[tuple[float, float, float, float]] = []
        for hand in result.hand_landmarks or []:
            xs = [min(max(point.x, 0.0), 1.0) for point in hand]
            ys = [min(max(point.y, 0.0), 1.0) for point in hand]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        return HandObservation(present=bool(boxes), boxes=boxes)

    def close(self) -> None:
        with self._lock:
            self._landmarker.close()


class HandDetectorFactory:
    def __init__(self, mode: str, model_path: Path) -> None:
        self.mode = mode
        self.model_path = model_path
        self.status = "disabled"
        if mode == "mediapipe":
            if not model_path.is_file():
                self.status = "model_missing"
                logger.warning("Hand landmarker file is missing; no_hands detection is disabled")
            else:
                try:
                    MediaPipeHandDetector(model_path).close()
                    self.status = "ready"
                except Exception as exc:  # mediapipe import / init failure
                    self.status = "unavailable"
                    logger.warning("MediaPipe hand detector unavailable: %s", type(exc).__name__)

    def create(self) -> HandDetector:
        if self.status == "ready":
            try:
                return MediaPipeHandDetector(self.model_path)
            except Exception as exc:
                logger.warning("Failed to create hand detector: %s", type(exc).__name__)
        return NullHandDetector()
