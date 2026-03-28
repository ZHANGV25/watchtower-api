"""Tests for Block 6: Anomaly Detection."""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from anomaly import AnomalyDetector, AnomalyPhase


class TestAnomalyPhase:
    def test_values(self):
        assert AnomalyPhase.OFF == "off"
        assert AnomalyPhase.LEARNING == "learning"
        assert AnomalyPhase.DETECTING == "detecting"


class TestAnomalyDetector:
    def test_init(self):
        ad = AnomalyDetector()
        assert ad.phase == AnomalyPhase.OFF
        assert ad.threshold == 0.35

    def test_start_learning(self):
        ad = AnomalyDetector()
        ad.start_learning()
        assert ad.phase == AnomalyPhase.LEARNING
        assert ad.learning_time_remaining > 0

    def test_stop(self):
        ad = AnomalyDetector()
        ad.start_learning()
        ad.stop()
        assert ad.phase == AnomalyPhase.OFF
        assert ad.learning_time_remaining == 0

    def test_threshold_bounds(self):
        ad = AnomalyDetector()
        ad.threshold = 0.01
        assert ad.threshold == 0.05  # Min bound

        ad.threshold = 1.5
        assert ad.threshold == 0.95  # Max bound

        ad.threshold = 0.5
        assert ad.threshold == 0.5

    def test_detect_when_off(self):
        ad = AnomalyDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        score = ad.detect(frame)
        assert score == 0.0

    def test_compute_features(self):
        ad = AnomalyDetector()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        features = ad._compute_features(frame)
        assert features.ndim == 1
        assert len(features) > 0
        # Features should be normalized (roughly between 0-1)
        assert features.max() <= 1.0 + 1e-6

    def test_learning_collects_baselines(self):
        ad = AnomalyDetector(learning_duration=0.1)  # Very short for testing
        ad.start_learning()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ad.learn_frame(frame)
        assert len(ad._baseline_histograms) == 1

        ad.learn_frame(frame)
        assert len(ad._baseline_histograms) == 2

    def test_learning_completes(self):
        ad = AnomalyDetector(learning_duration=0.01)  # Instant
        ad.start_learning()

        # Wait just enough
        time.sleep(0.02)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        done = ad.learn_frame(frame)

        assert done is True
        assert ad.phase == AnomalyPhase.DETECTING

    def test_detection_normal_scene(self):
        """Same scene as baseline should score low."""
        ad = AnomalyDetector(learning_duration=0.01)
        ad.start_learning()

        # Use consistent frame for baseline
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)

        # Feed baseline frames
        for _ in range(5):
            ad.learn_frame(frame)

        # Force finalize
        time.sleep(0.02)
        ad.learn_frame(frame)

        assert ad.phase == AnomalyPhase.DETECTING

        # Same frame should have low anomaly score
        score = ad.detect(frame)
        assert score < 0.3, f"Expected low score for normal scene, got {score}"

    def test_detection_anomalous_scene(self):
        """Very different scene should score higher."""
        ad = AnomalyDetector(learning_duration=0.01)
        ad.start_learning()

        # Baseline: dark frame
        baseline = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            ad.learn_frame(baseline)

        time.sleep(0.02)
        ad.learn_frame(baseline)
        assert ad.phase == AnomalyPhase.DETECTING

        # Anomalous: bright frame with different color distribution
        anomalous = np.full((480, 640, 3), 255, dtype=np.uint8)
        # Add colorful patches
        anomalous[100:300, 200:400] = [0, 0, 255]  # Red patch
        anomalous[300:400, 100:300] = [0, 255, 0]  # Green patch

        normal_score = ad.detect(baseline)
        anomaly_score = ad.detect(anomalous)

        assert anomaly_score > normal_score, (
            f"Anomalous score ({anomaly_score}) should be higher than normal ({normal_score})"
        )

    def test_learning_time_remaining_decreases(self):
        ad = AnomalyDetector(learning_duration=5.0)
        ad.start_learning()
        remaining = ad.learning_time_remaining
        assert 4.0 < remaining <= 5.0

    def test_multiple_start_stop_cycles(self):
        ad = AnomalyDetector(learning_duration=0.01)

        # First cycle
        ad.start_learning()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            ad.learn_frame(frame)
        time.sleep(0.02)
        ad.learn_frame(frame)
        assert ad.phase == AnomalyPhase.DETECTING

        # Stop and restart
        ad.stop()
        assert ad.phase == AnomalyPhase.OFF
        assert len(ad._baseline_histograms) == 0

        # Second cycle
        ad.start_learning()
        assert ad.phase == AnomalyPhase.LEARNING

    def test_baseline_frames_limit(self):
        ad = AnomalyDetector(learning_duration=100)  # Long duration
        ad.start_learning()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(100):
            ad.learn_frame(frame)

        assert len(ad._baseline_frames) <= ad._max_baseline_frames
