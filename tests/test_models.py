"""Tests for Pydantic data models."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import Alert, BBox, Condition, Detection, PoseKeypoint, Rule, WSMessage, Zone


class TestZone:
    def test_create_with_defaults(self):
        z = Zone(name="Door", x=10, y=20, width=30, height=40)
        assert z.name == "Door"
        assert z.id  # auto-generated
        assert z.color == "#22d3ee"

    def test_create_with_all_fields(self):
        z = Zone(id="abc", name="Window", x=0, y=0, width=100, height=100, color="#ff0000")
        assert z.id == "abc"
        assert z.color == "#ff0000"

    def test_serialization_roundtrip(self):
        z = Zone(name="Kitchen", x=50, y=50, width=25, height=25)
        data = z.model_dump()
        z2 = Zone.model_validate(data)
        assert z2.name == z.name
        assert z2.x == z.x


class TestDetection:
    def test_detection_without_pose(self):
        d = Detection(
            class_name="person",
            confidence=0.95,
            bbox=BBox(x=10, y=20, width=30, height=40),
        )
        assert d.pose is None

    def test_detection_with_pose(self):
        d = Detection(
            class_name="person",
            confidence=0.85,
            bbox=BBox(x=10, y=20, width=30, height=40),
            pose=[PoseKeypoint(name="nose", x=25, y=15, visibility=0.99)],
        )
        assert len(d.pose) == 1
        assert d.pose[0].name == "nose"


class TestRule:
    def test_rule_defaults(self):
        r = Rule(
            name="Person detector",
            natural_language="Alert if person detected",
            conditions=[Condition(type="object_present", params={"class": "person"})],
        )
        assert r.enabled is True
        assert r.severity == "medium"
        assert r.created_at > 0

    def test_rule_serialization(self):
        r = Rule(
            name="Test",
            natural_language="test",
            conditions=[Condition(type="object_present", params={"class": "cat"})],
            severity="high",
        )
        data = r.model_dump()
        assert data["severity"] == "high"
        assert data["conditions"][0]["type"] == "object_present"


class TestAlert:
    def test_alert_defaults(self):
        a = Alert(rule_id="r1", rule_name="Test rule", severity="low")
        assert a.frame_b64 == ""
        assert a.narration == ""
        assert a.detections == []


class TestWSMessage:
    def test_envelope(self):
        msg = WSMessage(type="frame", payload={"fps": 15})
        j = msg.model_dump_json()
        assert '"type":"frame"' in j or '"type": "frame"' in j
