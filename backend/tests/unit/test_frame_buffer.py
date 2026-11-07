from __future__ import annotations

import numpy as np

from app.recognition.frame_buffer import DropOldestQueue, FrameBuffer


def test_ring_buffer_keeps_last_frames_in_order() -> None:
    buffer = FrameBuffer(3)
    for value in range(5):
        buffer.append(np.full((1,), value))
    assert buffer.is_full and len(buffer) == 3
    assert [int(f[0]) for f in buffer.snapshot()] == [2, 3, 4]


def test_stride_counts_new_frames_since_last_inference() -> None:
    buffer = FrameBuffer(2)
    buffer.append(np.zeros(1))
    assert not buffer.ready_for_inference(1)
    buffer.append(np.zeros(1))
    assert buffer.ready_for_inference(2)
    buffer.snapshot()
    buffer.append(np.zeros(1))
    assert not buffer.ready_for_inference(2)
    buffer.append(np.zeros(1))
    assert buffer.ready_for_inference(2)


def test_clear_releases_frames() -> None:
    buffer = FrameBuffer(2)
    buffer.append(np.zeros(1))
    buffer.clear()
    assert len(buffer) == 0 and buffer.frames_since_inference == 0


def test_queue_drops_oldest_unprocessed_frame() -> None:
    queue = DropOldestQueue(2)
    for value in range(4):
        queue.put((bytes([value]), value))
    assert queue.dropped == 2
    assert queue.get() == (bytes([2]), 2) and queue.get() == (bytes([3]), 3) and queue.get() is None
