"""Utilities for processing YOLO segmentation masks into polygon points."""
from __future__ import annotations

import cv2
import numpy as np

from models import PolygonPoint

MAX_POLYGON_POINTS = 40


def extract_mask_polygon(
    masks: object | None,
    idx: int,
    frame_w: int,
    frame_h: int,
) -> list[PolygonPoint] | None:
    """Extract a segmentation mask polygon for the detection at the given index.

    Uses Douglas-Peucker polygon approximation for clean, smooth contours
    rather than raw point downsampling.

    Args:
        masks: YOLO masks object (has .xy attribute) or None.
        idx: Index of the detection to extract.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        List of PolygonPoint in percentage coordinates, or None.
    """
    if masks is None:
        return None

    xy_list = getattr(masks, "xy", None)
    if xy_list is None or idx >= len(xy_list):
        return None

    polygon_xy = xy_list[idx]
    if len(polygon_xy) < 3:
        return None

    # Douglas-Peucker simplification for clean contours
    contour = np.array(polygon_xy, dtype=np.float32).reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour, closed=True)
    # Adaptive epsilon: tighter for small objects, looser for large
    epsilon = perimeter * 0.015
    approx = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)

    # If still too many points, increase epsilon until under limit
    if len(approx) > MAX_POLYGON_POINTS:
        epsilon = perimeter * 0.025
        approx = cv2.approxPolyDP(contour, epsilon, closed=True).reshape(-1, 2)

    if len(approx) < 3:
        return None

    return [
        PolygonPoint(
            x=(float(pt[0]) / frame_w) * 100,
            y=(float(pt[1]) / frame_h) * 100,
        )
        for pt in approx
    ]
