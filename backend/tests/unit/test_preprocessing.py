from __future__ import annotations

import numpy as np
import pytest

from app.recognition.hand_detection import HandObservation, NullHandDetector
from app.recognition.preprocessing import (
    FramePreprocessor,
    FrameTooLargeError,
    InvalidFrameError,
    build_tensor,
    check_frame_bytes,
    crop_box,
    decode_jpeg,
    resize_image,
)
from app.schemas.manifest import Preprocessing
from tests.conftest import RED, jpeg


def config(**overrides: object) -> Preprocessing:
    values: dict[str, object] = {"width": 16, "height": 16, "normalization": "scale_0_1"}
    values.update(overrides)
    return Preprocessing.model_validate(values)


def test_decodes_valid_jpeg_to_bgr() -> None:
    image = decode_jpeg(jpeg(RED, (40, 30)))
    assert image.shape == (30, 40, 3)
    assert image[..., 2].mean() > 200 and image[..., 0].mean() < 40  # BGR order


@pytest.mark.parametrize("payload", [b"", b"PNG...." * 20, b"\xff\xd8\xff" + b"\x00" * 200])
def test_corrupted_or_non_jpeg_is_rejected(payload: bytes) -> None:
    with pytest.raises(InvalidFrameError):
        check_frame_bytes(payload, 1_000_000)
        decode_jpeg(payload)


def test_truncated_jpeg_is_rejected() -> None:
    data = jpeg(RED, (64, 64))
    with pytest.raises(InvalidFrameError):
        decode_jpeg(data[: len(data) // 3])


def test_frame_size_limit() -> None:
    with pytest.raises(FrameTooLargeError):
        check_frame_bytes(b"\xff\xd8\xff" + b"0" * 5000, 1024)


@pytest.mark.parametrize("mode", ["stretch", "pad", "center_crop"])
def test_resize_modes_produce_target_shape(mode: str) -> None:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    assert resize_image(image, config(resize=mode, width=40, height=20)).shape == (20, 40, 3)


def test_pad_keeps_aspect_ratio_with_pad_value() -> None:
    image = np.full((10, 40, 3), 255, dtype=np.uint8)
    out = resize_image(image, config(resize="pad", pad_value=114))
    assert (out[0] == 114).all() and (out[8] == 255).all()


def test_crop_box_is_square_with_margin_and_clamped() -> None:
    hands = HandObservation(present=True, boxes=[(0.4, 0.4, 0.5, 0.6)])
    left, top, right, bottom = crop_box(100, 100, hands, margin=0.5)
    assert right - left == bottom - top == 40
    assert left >= 0 and right <= 100 and top >= 0 and bottom <= 100
    assert crop_box(100, 100, HandObservation(present=False), 0.5) is None


def test_missing_hand_detector_reports_unknown_presence() -> None:
    frame = FramePreprocessor(config(), NullHandDetector(), 1_000_000).process(jpeg(RED), 0)
    assert frame.hands.present is None
    assert frame.image.shape == (16, 16, 3)
    assert frame.image[..., 0].mean() > 200  # RGB order from manifest


def test_build_tensor_layouts_and_normalization() -> None:
    frames = [np.full((16, 16, 3), 255, dtype=np.uint8)] * 4
    ncthw = build_tensor(frames, config(layout="NCTHW"))
    ntchw = build_tensor(frames, config(layout="NTCHW"))
    assert ncthw.shape == (1, 3, 4, 16, 16) and ntchw.shape == (1, 4, 3, 16, 16)
    assert np.allclose(ncthw, 1.0) and ncthw.dtype == np.float32
    imagenet = build_tensor(frames, config(normalization="imagenet"))
    assert np.isclose(imagenet[0, 0, 0, 0, 0], (255 - 123.675) / 58.395, atol=1e-4)
