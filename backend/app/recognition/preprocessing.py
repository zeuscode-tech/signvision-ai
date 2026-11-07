"""Frame preprocessing. All parameters come from the model manifest.

Per frame: validate size/format → decode JPEG → find hands → crop (if the
manifest asks for it) → resize → colour conversion. Normalisation and axis
layout are applied when the window tensor is built.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..schemas.manifest import Preprocessing
from .hand_detection import HandDetector, HandObservation

JPEG_MAGIC = b"\xff\xd8\xff"
MIN_SIDE = 16
MAX_SIDE = 4096
IMAGENET_MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


class InvalidFrameError(ValueError):
    pass


class FrameTooLargeError(ValueError):
    pass


@dataclass
class ProcessedFrame:
    image: np.ndarray  # H x W x 3 uint8 in the manifest colour order
    hands: HandObservation


def check_frame_bytes(data: bytes, max_bytes: int) -> None:
    if len(data) > max_bytes:
        raise FrameTooLargeError(f"Кадр {len(data)} байт превышает лимит {max_bytes}")
    if len(data) < 64 or not data.startswith(JPEG_MAGIC):
        raise InvalidFrameError("Ожидается JPEG-кадр")


def decode_jpeg(data: bytes) -> np.ndarray:
    """Decode JPEG bytes into a BGR uint8 image."""
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise InvalidFrameError("Не удалось декодировать JPEG")
    height, width = image.shape[:2]
    if min(height, width) < MIN_SIDE or max(height, width) > MAX_SIDE:
        raise InvalidFrameError("Недопустимый размер кадра")
    return image


def crop_box(
    width: int, height: int, hands: HandObservation, margin: float
) -> tuple[int, int, int, int] | None:
    """Square crop around all detected hands with a safety margin, clamped to the frame."""
    if not hands.boxes:
        return None
    x0 = min(box[0] for box in hands.boxes) * width
    y0 = min(box[1] for box in hands.boxes) * height
    x1 = max(box[2] for box in hands.boxes) * width
    y1 = max(box[3] for box in hands.boxes) * height
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    side = min(max(side, MIN_SIDE), float(min(width, height)))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = int(round(min(max(cx - side / 2, 0), width - side)))
    top = int(round(min(max(cy - side / 2, 0), height - side)))
    size = int(round(side))
    return left, top, left + size, top + size


def resize_image(image: np.ndarray, config: Preprocessing) -> np.ndarray:
    target_w, target_h = config.width, config.height
    height, width = image.shape[:2]
    if config.resize == "stretch":
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if config.resize == "center_crop":
        scale = max(target_w / width, target_h / height)
        resized = cv2.resize(
            image,
            (max(target_w, round(width * scale)), max(target_h, round(height * scale))),
            interpolation=cv2.INTER_LINEAR,
        )
        top = (resized.shape[0] - target_h) // 2
        left = (resized.shape[1] - target_w) // 2
        return resized[top : top + target_h, left : left + target_w]
    # pad: keep aspect ratio, letterbox with pad_value
    scale = min(target_w / width, target_h / height)
    new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), config.pad_value, dtype=np.uint8)
    top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


class FramePreprocessor:
    def __init__(self, config: Preprocessing, detector: HandDetector, max_frame_bytes: int) -> None:
        self.config = config
        self.detector = detector
        self.max_frame_bytes = max_frame_bytes

    def process(self, data: bytes, timestamp_ms: int) -> ProcessedFrame:
        check_frame_bytes(data, self.max_frame_bytes)
        bgr = decode_jpeg(data)
        return self.process_image(bgr, timestamp_ms)

    def process_image(self, bgr: np.ndarray, timestamp_ms: int) -> ProcessedFrame:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hands = self.detector.detect(rgb, timestamp_ms)
        image = rgb if self.config.color == "RGB" else bgr
        if self.config.crop == "hands":
            box = crop_box(image.shape[1], image.shape[0], hands, self.config.crop_margin)
            if box is not None:
                left, top, right, bottom = box
                image = image[top:bottom, left:right]
        return ProcessedFrame(image=np.ascontiguousarray(resize_image(image, self.config)), hands=hands)


def build_tensor(frames: list[np.ndarray], config: Preprocessing) -> np.ndarray:
    """Stack T frames (H x W x C uint8) into the model's 5-D float32 input tensor."""
    clip = np.stack(frames, axis=0).astype(np.float32)  # T, H, W, C
    if config.normalization == "scale_0_1":
        clip /= 255.0
    elif config.normalization == "imagenet":
        clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
    elif config.normalization == "custom":
        assert config.mean is not None and config.std is not None
        clip = (clip - np.asarray(config.mean, dtype=np.float32)) / np.asarray(config.std, dtype=np.float32)
    if config.layout == "NCTHW":
        tensor = np.transpose(clip, (3, 0, 1, 2))  # C, T, H, W
    else:
        tensor = np.transpose(clip, (0, 3, 1, 2))  # T, C, H, W
    return np.ascontiguousarray(tensor[None], dtype=np.float32)
