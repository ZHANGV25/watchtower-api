from __future__ import annotations

import threading
from collections import deque

import cv2
import numpy as np


class ReplayBuffer:
    """Circular buffer storing recent frames at reduced resolution."""

    def __init__(self, max_seconds: int = 1800, fps: int = 2) -> None:
        self._max_frames = max_seconds * fps
        self._target_fps = fps
        self._frames: deque[tuple[np.ndarray, float]] = deque(maxlen=self._max_frames)
        self._lock = threading.Lock()
        self._last_add_time: float = 0.0
        self._frame_interval = 1.0 / fps
        self._resize_width = 640

    def add_frame(self, frame: np.ndarray, timestamp: float) -> None:
        if timestamp - self._last_add_time < self._frame_interval:
            return

        h, w = frame.shape[:2]
        if w > self._resize_width:
            scale = self._resize_width / w
            small = cv2.resize(
                frame,
                (self._resize_width, int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame.copy()

        with self._lock:
            self._frames.append((small, timestamp))
        self._last_add_time = timestamp

    def get_frames(
        self,
        start_time: float,
        duration: float = 10.0,
    ) -> list[tuple[np.ndarray, float]]:
        end_time = start_time + duration
        with self._lock:
            return [
                (f, t) for f, t in self._frames
                if start_time <= t <= end_time
            ]

    def get_timestamps(self) -> list[float]:
        with self._lock:
            return [t for _, t in self._frames]

    def get_frame_at(self, timestamp: float) -> tuple[np.ndarray, float] | None:
        with self._lock:
            best = None
            best_diff = float("inf")
            for frame, t in self._frames:
                diff = abs(t - timestamp)
                if diff < best_diff:
                    best_diff = diff
                    best = (frame, t)
            return best
