from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


def _uuid() -> str:
    return uuid.uuid4().hex[:8]


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------

class Zone(BaseModel):
    id: str = Field(default_factory=_uuid)
    name: str
    x: float           # percentage 0-100
    y: float
    width: float
    height: float
    color: str = "#22d3ee"  # cyan-400 default


# ---------------------------------------------------------------------------
# Detection primitives
# ---------------------------------------------------------------------------

class BBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class PoseKeypoint(BaseModel):
    name: str
    x: float
    y: float
    visibility: float


class PolygonPoint(BaseModel):
    x: float  # percentage 0-100
    y: float


class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: BBox
    pose: Optional[list[PoseKeypoint]] = None
    mask: Optional[list[PolygonPoint]] = None  # segmentation polygon outline


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

class Condition(BaseModel):
    type: str   # object_present, object_in_zone, object_absent, person_size,
                # person_pose, person_falling, count, duration,
                # time_window, movement_speed, stillness
    params: dict[str, Any] = Field(default_factory=dict)


class Rule(BaseModel):
    id: str = Field(default_factory=_uuid)
    name: str
    natural_language: str
    conditions: list[Condition]
    severity: str = "medium"   # low | medium | high | critical
    enabled: bool = True
    created_at: float = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Monitoring Plans
# ---------------------------------------------------------------------------

class MonitoringPlan(BaseModel):
    id: str = Field(default_factory=_uuid)
    name: str
    description: str
    scenario: str
    rules: list[Rule]
    zones: list[Zone] = Field(default_factory=list)
    created_at: float = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

class Alert(BaseModel):
    id: str = Field(default_factory=_uuid)
    rule_id: str
    rule_name: str
    severity: str
    timestamp: float = Field(default_factory=_now)
    frame_b64: str = ""
    narration: str = ""
    detections: list[Detection] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# WebSocket message envelope
# ---------------------------------------------------------------------------

class WSMessage(BaseModel):
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
