"""Block 6: Anomaly detection via frame embeddings.

"Just Watch" mode: learn what normal looks like for 2 minutes, then
alert when something unusual appears — without any explicit rules.

Uses simple frame differencing as a fallback when Gemini isn't available.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

log = logging.getLogger("watchtower.anomaly")


class AnomalyPhase(str, Enum):
    OFF = "off"
    LEARNING = "learning"
    DETECTING = "detecting"


@dataclass
class AnomalyResult:
    score: float  # 0.0 = normal, 1.0 = very anomalous
    description: str = ""


class AnomalyDetector:
    def __init__(self, learning_duration: float = 120.0, threshold: float = 0.35) -> None:
        self._phase = AnomalyPhase.OFF
        self._learning_duration = learning_duration  # seconds
        self._threshold = threshold
        self._learning_start: float = 0.0

        # Baseline frame features (simple approach: grayscale histograms)
        self._baseline_histograms: list[np.ndarray] = []
        self._baseline_mean: np.ndarray | None = None
        self._baseline_std: np.ndarray | None = None

        # For structural similarity + LLM comparison (stored in color at 480x360)
        self._baseline_frames: list[np.ndarray] = []
        self._max_baseline_frames = 20  # fewer frames, higher quality

    @property
    def phase(self) -> AnomalyPhase:
        return self._phase

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = max(0.05, min(0.95, value))

    @property
    def learning_time_remaining(self) -> float:
        if self._phase != AnomalyPhase.LEARNING:
            return 0.0
        elapsed = time.time() - self._learning_start
        return max(0.0, self._learning_duration - elapsed)

    def start_learning(self) -> None:
        """Begin the learning phase."""
        self._phase = AnomalyPhase.LEARNING
        self._learning_start = time.time()
        self._baseline_histograms = []
        self._baseline_frames = []
        self._baseline_mean = None
        self._baseline_std = None
        log.info("Anomaly detection: learning phase started (%.0fs)", self._learning_duration)

    def stop(self) -> None:
        """Stop anomaly detection."""
        self._phase = AnomalyPhase.OFF
        self._baseline_histograms = []
        self._baseline_frames = []
        self._baseline_mean = None
        self._baseline_std = None
        log.info("Anomaly detection: stopped")

    def _compute_features(self, frame: np.ndarray) -> np.ndarray:
        """Compute a feature vector from a frame using color + edge histograms."""
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Color histogram (HSV)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()

        # Edge histogram (gradient magnitude)
        edges = cv2.Canny(gray, 50, 150)
        # Divide into 4x3 grid and count edge pixels per cell
        h, w = edges.shape
        grid_h, grid_w = h // 3, w // 4
        edge_features: list[float] = []
        for gy in range(3):
            for gx in range(4):
                cell = edges[gy * grid_h:(gy + 1) * grid_h, gx * grid_w:(gx + 1) * grid_w]
                edge_features.append(float(np.sum(cell > 0)))

        # Combine and normalize
        features = np.concatenate([
            hist_h / (hist_h.sum() + 1e-6),
            hist_s / (hist_s.sum() + 1e-6),
            np.array(edge_features) / (max(edge_features) + 1e-6),
        ])
        return features

    def learn_frame(self, frame: np.ndarray) -> bool:
        """Process a frame during the learning phase. Returns True when learning is complete."""
        if self._phase != AnomalyPhase.LEARNING:
            return False

        features = self._compute_features(frame)
        self._baseline_histograms.append(features)

        # Store color frame for LLM comparison + structural similarity
        if len(self._baseline_frames) < self._max_baseline_frames:
            small = cv2.resize(frame, (480, 360))
            self._baseline_frames.append(small)

        # Check if learning period is over
        if time.time() - self._learning_start >= self._learning_duration:
            self._finalize_learning()
            return True
        return False

    def _finalize_learning(self) -> None:
        """Compute baseline statistics from collected features."""
        if not self._baseline_histograms:
            self._phase = AnomalyPhase.OFF
            return

        all_features = np.array(self._baseline_histograms)
        self._baseline_mean = np.mean(all_features, axis=0)
        self._baseline_std = np.std(all_features, axis=0) + 1e-6  # avoid division by zero

        self._phase = AnomalyPhase.DETECTING
        log.info(
            "Anomaly detection: learning complete (%d samples), now detecting",
            len(self._baseline_histograms),
        )

    def detect(self, frame: np.ndarray) -> float:
        """Detect anomaly in a frame. Returns score 0.0 (normal) to 1.0 (anomalous)."""
        if self._phase != AnomalyPhase.DETECTING or self._baseline_mean is None:
            return 0.0

        features = self._compute_features(frame)

        # Z-score based anomaly: how many std devs from baseline mean
        z_scores = np.abs(features - self._baseline_mean) / self._baseline_std
        mean_z = float(np.mean(z_scores))

        # Also compute structural difference from baseline frames
        struct_diff = 0.0
        if self._baseline_frames:
            small = cv2.resize(frame, (480, 360))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            diffs = [
                float(np.mean(cv2.absdiff(gray, cv2.cvtColor(bf, cv2.COLOR_BGR2GRAY)))) / 255.0
                for bf in self._baseline_frames[::max(1, len(self._baseline_frames) // 5)]
            ]
            struct_diff = min(diffs)  # closest baseline frame

        # Combine: z-score (distribution) + structural difference
        # Normalize z-score: typical range 0-5, map to 0-1
        z_normalized = min(1.0, mean_z / 5.0)
        # Weight: 60% histogram, 40% structural
        score = 0.6 * z_normalized + 0.4 * struct_diff

        return min(1.0, max(0.0, score))
