"""Tests for detection models and mask extraction utilities."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

from models import BBox, Detection, PolygonPoint, PoseKeypoint
from mask_utils import extract_mask_polygon


# ---------------------------------------------------------------------------
# PolygonPoint model tests
# ---------------------------------------------------------------------------

class TestPolygonPoint:
    def test_create(self):
        p = PolygonPoint(x=50.0, y=75.0)
        assert p.x == 50.0
        assert p.y == 75.0

    def test_serialization(self):
        p = PolygonPoint(x=10.5, y=20.3)
        data = p.model_dump()
        assert data == {"x": 10.5, "y": 20.3}

    def test_roundtrip(self):
        p = PolygonPoint(x=33.33, y=66.66)
        p2 = PolygonPoint.model_validate(p.model_dump())
        assert p2.x == p.x
        assert p2.y == p.y


# ---------------------------------------------------------------------------
# Detection with mask tests
# ---------------------------------------------------------------------------

class TestDetectionWithMask:
    def test_detection_without_mask(self):
        d = Detection(
            class_name="person",
            confidence=0.9,
            bbox=BBox(x=10, y=20, width=30, height=40),
        )
        assert d.mask is None

    def test_detection_with_mask(self):
        mask = [
            PolygonPoint(x=10, y=20),
            PolygonPoint(x=30, y=20),
            PolygonPoint(x=30, y=60),
            PolygonPoint(x=10, y=60),
        ]
        d = Detection(
            class_name="person",
            confidence=0.85,
            bbox=BBox(x=10, y=20, width=20, height=40),
            mask=mask,
        )
        assert d.mask is not None
        assert len(d.mask) == 4
        assert d.mask[0].x == 10
        assert d.mask[0].y == 20

    def test_detection_with_empty_mask(self):
        d = Detection(
            class_name="chair",
            confidence=0.7,
            bbox=BBox(x=50, y=50, width=10, height=10),
            mask=[],
        )
        assert d.mask == []

    def test_detection_serialization_with_mask(self):
        mask = [
            PolygonPoint(x=5, y=10),
            PolygonPoint(x=25, y=10),
            PolygonPoint(x=25, y=50),
        ]
        d = Detection(
            class_name="dog",
            confidence=0.92,
            bbox=BBox(x=5, y=10, width=20, height=40),
            mask=mask,
        )
        data = d.model_dump()
        assert len(data["mask"]) == 3
        assert data["mask"][0] == {"x": 5.0, "y": 10.0}

    def test_detection_roundtrip_with_mask(self):
        mask = [PolygonPoint(x=i * 10, y=i * 5) for i in range(10)]
        d = Detection(
            class_name="person",
            confidence=0.88,
            bbox=BBox(x=0, y=0, width=50, height=80),
            mask=mask,
            pose=[PoseKeypoint(name="nose", x=25, y=10, visibility=0.99)],
        )
        data = d.model_dump()
        d2 = Detection.model_validate(data)
        assert len(d2.mask) == 10
        assert d2.mask[3].x == 30
        assert d2.pose is not None
        assert len(d2.pose) == 1


# ---------------------------------------------------------------------------
# Mock YOLO masks object
# ---------------------------------------------------------------------------

class MockMasks:
    """Mimics the ultralytics masks.xy structure."""

    def __init__(self, polygons: list[np.ndarray]):
        self.xy = polygons


# ---------------------------------------------------------------------------
# extract_mask_polygon tests
# ---------------------------------------------------------------------------

class TestExtractMaskPolygon:
    def test_none_masks(self):
        result = extract_mask_polygon(None, 0, 640, 480)
        assert result is None

    def test_empty_polygon(self):
        masks = MockMasks([np.array([])])
        result = extract_mask_polygon(masks, 0, 640, 480)
        assert result is None

    def test_index_out_of_range(self):
        masks = MockMasks([np.array([[100, 100], [200, 100], [200, 200]])])
        result = extract_mask_polygon(masks, 5, 640, 480)
        assert result is None

    def test_simple_triangle(self):
        polygon = np.array([
            [0, 0],
            [320, 0],
            [160, 240],
        ], dtype=np.float32)
        masks = MockMasks([polygon])
        result = extract_mask_polygon(masks, 0, 640, 480)

        assert result is not None
        assert len(result) == 3
        assert result[0].x == pytest.approx(0.0)
        assert result[0].y == pytest.approx(0.0)
        assert result[1].x == pytest.approx(50.0)
        assert result[1].y == pytest.approx(0.0)
        assert result[2].x == pytest.approx(25.0)
        assert result[2].y == pytest.approx(50.0)

    def test_simplification_reduces_points(self):
        """Complex polygon should be simplified via Douglas-Peucker."""
        # Circle-like polygon with 200 points -- should be simplified significantly
        angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        points = np.column_stack([
            500 + 300 * np.cos(angles),
            500 + 300 * np.sin(angles),
        ]).astype(np.float32)
        masks = MockMasks([points])
        result = extract_mask_polygon(masks, 0, 1000, 1000)

        assert result is not None
        assert len(result) < 200
        assert len(result) <= 40  # MAX_POLYGON_POINTS

    def test_simple_polygon_preserved(self):
        """A clean rectangle should stay roughly the same."""
        points = np.array([
            [100, 100], [500, 100], [500, 400], [100, 400],
        ], dtype=np.float32)
        masks = MockMasks([points])
        result = extract_mask_polygon(masks, 0, 1000, 1000)

        assert result is not None
        assert len(result) == 4

    def test_small_polygon_not_lost(self):
        """A valid small polygon should not be simplified to nothing."""
        points = np.array([
            [10, 10], [50, 10], [50, 50], [10, 50],
        ], dtype=np.float32)
        masks = MockMasks([points])
        result = extract_mask_polygon(masks, 0, 1000, 1000)

        assert result is not None
        assert len(result) >= 3

    def test_multiple_detections_correct_index(self):
        poly0 = np.array([[0, 0], [100, 0], [100, 100]], dtype=np.float32)
        poly1 = np.array([[200, 200], [300, 200], [300, 300], [200, 300]], dtype=np.float32)
        masks = MockMasks([poly0, poly1])

        result0 = extract_mask_polygon(masks, 0, 640, 480)
        result1 = extract_mask_polygon(masks, 1, 640, 480)

        assert result0 is not None
        assert len(result0) == 3
        assert result1 is not None
        assert len(result1) == 4

    def test_percentage_coordinates(self):
        """Verify output is in percentage coordinates, not pixel."""
        polygon = np.array([
            [0, 0], [640, 0], [640, 480], [0, 480],
        ], dtype=np.float32)
        masks = MockMasks([polygon])
        result = extract_mask_polygon(masks, 0, 640, 480)

        assert result is not None
        # All points should be in 0-100 range
        for pt in result:
            assert 0.0 <= pt.x <= 100.0
            assert 0.0 <= pt.y <= 100.0

    def test_masks_without_xy_attribute(self):
        """Object without .xy should return None."""
        result = extract_mask_polygon(object(), 0, 640, 480)
        assert result is None
