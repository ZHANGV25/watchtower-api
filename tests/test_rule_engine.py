"""Tests for the rule engine -- all condition types and edge cases."""
from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import BBox, Condition, Detection, PoseKeypoint, Rule, Zone
from rule_engine import RuleEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _person(x=10, y=10, w=20, h=50, conf=0.9, pose=None):
    return Detection(
        class_name="person",
        confidence=conf,
        bbox=BBox(x=x, y=y, width=w, height=h),
        pose=pose,
    )


def _object(cls, x=10, y=10, w=10, h=10, conf=0.8):
    return Detection(
        class_name=cls,
        confidence=conf,
        bbox=BBox(x=x, y=y, width=w, height=h),
    )


def _rule(name, conditions, severity="medium", enabled=True, rule_id=None):
    r = Rule(
        name=name,
        natural_language=name,
        conditions=[Condition(type=c[0], params=c[1]) for c in conditions],
        severity=severity,
        enabled=enabled,
    )
    if rule_id:
        r.id = rule_id
    return r


def _zone(name, x, y, w, h):
    return Zone(name=name, x=x, y=y, width=w, height=h)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestObjectPresent:
    def test_person_detected(self):
        engine = RuleEngine()
        rule = _rule("detect person", [("object_present", {"class": "person"})])
        alerts = engine.evaluate([rule], [], [_person()], time.time())
        assert len(alerts) == 1
        assert alerts[0].rule_name == "detect person"

    def test_no_person(self):
        engine = RuleEngine()
        rule = _rule("detect person", [("object_present", {"class": "person"})])
        alerts = engine.evaluate([rule], [], [_object("cat")], time.time())
        assert len(alerts) == 0

    def test_specific_class(self):
        engine = RuleEngine()
        rule = _rule("detect dog", [("object_present", {"class": "dog"})])
        alerts = engine.evaluate([rule], [], [_object("dog")], time.time())
        assert len(alerts) == 1


class TestObjectAbsent:
    def test_object_missing(self):
        engine = RuleEngine()
        rule = _rule("no person", [("object_absent", {"class": "person"})])
        alerts = engine.evaluate([rule], [], [_object("cat")], time.time())
        assert len(alerts) == 1

    def test_object_present_no_alert(self):
        engine = RuleEngine()
        rule = _rule("no person", [("object_absent", {"class": "person"})])
        alerts = engine.evaluate([rule], [], [_person()], time.time())
        assert len(alerts) == 0


class TestObjectInZone:
    def test_person_in_zone(self):
        engine = RuleEngine()
        zone = _zone("Door", 0, 0, 50, 50)
        # Person bbox center = 10 + 20/2 = 20, 10 + 50/2 = 35 -> inside zone
        rule = _rule("person at door", [("object_in_zone", {"class": "person", "zone": "Door"})])
        alerts = engine.evaluate([rule], [zone], [_person(x=10, y=10, w=20, h=50)], time.time())
        assert len(alerts) == 1

    def test_person_outside_zone(self):
        engine = RuleEngine()
        zone = _zone("Door", 0, 0, 10, 10)
        # Person center at 20, 35 -> outside zone (0-10, 0-10)
        rule = _rule("person at door", [("object_in_zone", {"class": "person", "zone": "Door"})])
        alerts = engine.evaluate([rule], [zone], [_person()], time.time())
        assert len(alerts) == 0

    def test_zone_name_case_insensitive(self):
        engine = RuleEngine()
        zone = _zone("Kitchen Counter", 0, 0, 100, 100)
        rule = _rule("in kitchen", [("object_in_zone", {"class": "person", "zone": "kitchen counter"})])
        alerts = engine.evaluate([rule], [zone], [_person()], time.time())
        assert len(alerts) == 1

    def test_unknown_zone(self):
        engine = RuleEngine()
        rule = _rule("in nowhere", [("object_in_zone", {"class": "person", "zone": "NonExistent"})])
        alerts = engine.evaluate([rule], [], [_person()], time.time())
        assert len(alerts) == 0


class TestCount:
    def test_count_gte(self):
        engine = RuleEngine()
        rule = _rule("crowd", [("count", {"class": "person", "operator": "gte", "value": 2})])
        alerts = engine.evaluate([rule], [], [_person(x=10), _person(x=50)], time.time())
        assert len(alerts) == 1

    def test_count_lt(self):
        engine = RuleEngine()
        rule = _rule("few people", [("count", {"class": "person", "operator": "lt", "value": 3})])
        alerts = engine.evaluate([rule], [], [_person()], time.time())
        assert len(alerts) == 1

    def test_count_eq(self):
        engine = RuleEngine()
        rule = _rule("exactly two", [("count", {"class": "person", "operator": "eq", "value": 2})])
        alerts = engine.evaluate([rule], [], [_person()], time.time())
        assert len(alerts) == 0


class TestPersonSize:
    def test_small_person(self):
        engine = RuleEngine()
        # height = 20 -> ratio = 20/100 = 0.2 < 0.3 threshold -> small
        rule = _rule("child", [("person_size", {"size": "small", "threshold": 0.3})])
        alerts = engine.evaluate([rule], [], [_person(h=20)], time.time())
        assert len(alerts) == 1

    def test_large_person(self):
        engine = RuleEngine()
        # height = 50 -> ratio = 0.5 >= 0.3 -> large
        rule = _rule("adult", [("person_size", {"size": "large", "threshold": 0.3})])
        alerts = engine.evaluate([rule], [], [_person(h=50)], time.time())
        assert len(alerts) == 1


class TestPersonPose:
    def test_standing(self):
        engine = RuleEngine()
        pose = [
            PoseKeypoint(name="left_shoulder", x=40, y=20, visibility=0.9),
            PoseKeypoint(name="right_shoulder", x=45, y=20, visibility=0.9),
            PoseKeypoint(name="left_hip", x=40, y=50, visibility=0.9),
            PoseKeypoint(name="right_hip", x=45, y=50, visibility=0.9),
            PoseKeypoint(name="left_knee", x=40, y=75, visibility=0.9),
            PoseKeypoint(name="right_knee", x=45, y=75, visibility=0.9),
        ]
        rule = _rule("standing", [("person_pose", {"pose": "standing"})])
        alerts = engine.evaluate([rule], [], [_person(pose=pose)], time.time())
        assert len(alerts) == 1

    def test_lying(self):
        engine = RuleEngine()
        # For lying: shoulder midpoint and hip midpoint must have large x-gap, small y-gap
        # shoulder midpoint = (22.5, 50), hip midpoint = (62.5, 51)
        # torso_width = 40, torso_height = 1 -> 40 > 1*1.5 -> lying
        pose = [
            PoseKeypoint(name="left_shoulder", x=20, y=50, visibility=0.9),
            PoseKeypoint(name="right_shoulder", x=25, y=50, visibility=0.9),
            PoseKeypoint(name="left_hip", x=60, y=51, visibility=0.9),
            PoseKeypoint(name="right_hip", x=65, y=51, visibility=0.9),
        ]
        rule = _rule("lying", [("person_pose", {"pose": "lying"})])
        alerts = engine.evaluate([rule], [], [_person(pose=pose)], time.time())
        assert len(alerts) == 1


class TestDuration:
    def test_not_met_immediately(self):
        engine = RuleEngine()
        now = time.time()
        rule = _rule("lingering", [
            ("object_present", {"class": "person"}),
            ("duration", {"seconds": 5}),
        ], rule_id="dur1")
        # First eval: duration tracking starts, not met yet
        alerts = engine.evaluate([rule], [], [_person()], now)
        assert len(alerts) == 0

    def test_met_after_time(self):
        engine = RuleEngine()
        now = time.time()
        rule = _rule("lingering", [
            ("object_present", {"class": "person"}),
            ("duration", {"seconds": 2}),
        ], rule_id="dur2")
        # First call starts tracking
        engine.evaluate([rule], [], [_person()], now)
        # After 3 seconds
        alerts = engine.evaluate([rule], [], [_person()], now + 3)
        assert len(alerts) == 1


class TestCooldown:
    def test_no_double_fire(self):
        engine = RuleEngine()
        now = time.time()
        rule = _rule("detect", [("object_present", {"class": "person"})], rule_id="cd1")
        alerts1 = engine.evaluate([rule], [], [_person()], now)
        assert len(alerts1) == 1
        # Immediately after: cooldown should prevent re-fire
        alerts2 = engine.evaluate([rule], [], [_person()], now + 1)
        assert len(alerts2) == 0

    def test_fires_after_cooldown(self):
        engine = RuleEngine()
        now = time.time()
        rule = _rule("detect", [("object_present", {"class": "person"})], rule_id="cd2")
        engine.evaluate([rule], [], [_person()], now)
        # After cooldown (15s)
        alerts = engine.evaluate([rule], [], [_person()], now + 16)
        assert len(alerts) == 1


class TestDisabledRule:
    def test_disabled_rule_skipped(self):
        engine = RuleEngine()
        rule = _rule("detect", [("object_present", {"class": "person"})], enabled=False)
        alerts = engine.evaluate([rule], [], [_person()], time.time())
        assert len(alerts) == 0


class TestMultipleConditions:
    def test_all_conditions_must_match(self):
        engine = RuleEngine()
        rule = _rule("person and cat", [
            ("object_present", {"class": "person"}),
            ("object_present", {"class": "cat"}),
        ])
        # Only person, no cat
        alerts = engine.evaluate([rule], [], [_person()], time.time())
        assert len(alerts) == 0

    def test_all_conditions_met(self):
        engine = RuleEngine()
        rule = _rule("person and cat", [
            ("object_present", {"class": "person"}),
            ("object_present", {"class": "cat"}),
        ])
        alerts = engine.evaluate([rule], [], [_person(), _object("cat")], time.time())
        assert len(alerts) == 1


class TestMultipleRules:
    def test_independent_rules(self):
        engine = RuleEngine()
        now = time.time()
        rule1 = _rule("person", [("object_present", {"class": "person"})], rule_id="r1")
        rule2 = _rule("cat", [("object_present", {"class": "cat"})], rule_id="r2")
        alerts = engine.evaluate([rule1, rule2], [], [_person(), _object("cat")], now)
        assert len(alerts) == 2
