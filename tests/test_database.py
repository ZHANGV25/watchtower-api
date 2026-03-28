"""Tests for database CRUD operations."""
from __future__ import annotations

import asyncio
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Use temp DB for tests
os.environ["WATCHTOWER_DB"] = "/tmp/watchtower_test.db"

import database as db
from models import Alert, Camera, Condition, Rule, User, Zone, MemoryEntry


@pytest.fixture(autouse=True)
def _clean_db():
    """Re-initialize DB before each test."""
    import asyncio
    # Remove old DB
    try:
        os.remove("/tmp/watchtower_test.db")
    except FileNotFoundError:
        pass
    db._db = None
    yield
    asyncio.get_event_loop().run_until_complete(db.close_db())


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestCameras:
    def test_create_and_get(self):
        cam = Camera(name="Front Door", location="Entrance")
        run(db.create_camera(cam))
        result = run(db.get_camera(cam.id))
        assert result is not None
        assert result.name == "Front Door"
        assert result.location == "Entrance"

    def test_list_cameras(self):
        run(db.create_camera(Camera(name="Cam 1")))
        run(db.create_camera(Camera(name="Cam 2")))
        cameras = run(db.list_cameras())
        assert len(cameras) == 2

    def test_update_camera(self):
        cam = Camera(name="Old Name")
        run(db.create_camera(cam))
        run(db.update_camera(cam.id, name="New Name"))
        result = run(db.get_camera(cam.id))
        assert result.name == "New Name"

    def test_delete_camera(self):
        cam = Camera(name="To Delete")
        run(db.create_camera(cam))
        deleted = run(db.delete_camera(cam.id))
        assert deleted is True
        result = run(db.get_camera(cam.id))
        assert result is None

    def test_delete_nonexistent(self):
        deleted = run(db.delete_camera("nope"))
        assert deleted is False

    def test_heartbeat(self):
        cam = Camera(name="Heartbeat Test")
        run(db.create_camera(cam))
        run(db.camera_heartbeat(cam.id))
        result = run(db.get_camera(cam.id))
        assert result.status == "online"
        assert result.last_seen > 0

    def test_offline(self):
        cam = Camera(name="Offline Test", status="online")
        run(db.create_camera(cam))
        run(db.camera_offline(cam.id))
        result = run(db.get_camera(cam.id))
        assert result.status == "offline"


class TestZones:
    def test_create_and_list(self):
        cam = Camera(name="Test Cam")
        run(db.create_camera(cam))
        zone = Zone(camera_id=cam.id, name="Door", x=10, y=20, width=30, height=40)
        run(db.create_zone(zone))
        zones = run(db.list_zones(cam.id))
        assert len(zones) == 1
        assert zones[0].name == "Door"

    def test_replace_zones(self):
        cam = Camera(name="Test")
        run(db.create_camera(cam))
        run(db.create_zone(Zone(camera_id=cam.id, name="Old", x=0, y=0, width=10, height=10)))
        new_zones = [
            Zone(camera_id=cam.id, name="New1", x=0, y=0, width=50, height=50),
            Zone(camera_id=cam.id, name="New2", x=50, y=50, width=50, height=50),
        ]
        run(db.replace_zones(cam.id, new_zones))
        zones = run(db.list_zones(cam.id))
        assert len(zones) == 2
        names = {z.name for z in zones}
        assert "Old" not in names
        assert "New1" in names

    def test_delete_zone(self):
        cam = Camera(name="Test")
        run(db.create_camera(cam))
        zone = Zone(camera_id=cam.id, name="To Delete", x=0, y=0, width=10, height=10)
        run(db.create_zone(zone))
        deleted = run(db.delete_zone(zone.id))
        assert deleted is True
        assert len(run(db.list_zones(cam.id))) == 0

    def test_cascade_delete(self):
        cam = Camera(name="Cascade")
        run(db.create_camera(cam))
        run(db.create_zone(Zone(camera_id=cam.id, name="Z1", x=0, y=0, width=10, height=10)))
        run(db.delete_camera(cam.id))
        zones = run(db.list_zones(cam.id))
        assert len(zones) == 0


class TestRules:
    def test_create_and_list(self):
        cam = Camera(name="Test")
        run(db.create_camera(cam))
        rule = Rule(camera_id=cam.id, name="Fall", natural_language="Detect falls",
                    conditions=[Condition(type="person_pose", params={"pose": "lying"})], severity="critical")
        run(db.create_rule(rule))
        rules = run(db.list_rules(cam.id))
        assert len(rules) == 1
        assert rules[0].name == "Fall"
        assert rules[0].conditions[0].type == "person_pose"
        assert rules[0].enabled is True

    def test_update_rule(self):
        cam = Camera(name="Test")
        run(db.create_camera(cam))
        rule = Rule(camera_id=cam.id, name="R1", natural_language="test", conditions=[])
        run(db.create_rule(rule))
        run(db.update_rule(rule.id, enabled=False))
        rules = run(db.list_rules(cam.id))
        assert rules[0].enabled is False

    def test_delete_rules_for_camera(self):
        cam = Camera(name="Test")
        run(db.create_camera(cam))
        run(db.create_rule(Rule(camera_id=cam.id, name="R1", natural_language="t", conditions=[])))
        run(db.create_rule(Rule(camera_id=cam.id, name="R2", natural_language="t", conditions=[])))
        run(db.delete_rules_for_camera(cam.id))
        assert len(run(db.list_rules(cam.id))) == 0


class TestAlerts:
    def test_create_and_list(self):
        cam = Camera(name="Test")
        run(db.create_camera(cam))
        alert = Alert(camera_id=cam.id, rule_id="r1", rule_name="Fall", severity="high")
        run(db.create_alert(alert, frame_path="frames/test.jpg"))
        alerts = run(db.list_alerts(cam.id))
        assert len(alerts) == 1
        assert alerts[0]["rule_name"] == "Fall"
        assert alerts[0]["frame_path"] == "frames/test.jpg"

    def test_count_alerts(self):
        cam = Camera(name="Test")
        run(db.create_camera(cam))
        for i in range(5):
            run(db.create_alert(Alert(camera_id=cam.id, rule_id="r1", rule_name=f"A{i}", severity="low")))
        assert run(db.count_alerts(cam.id)) == 5

    def test_pagination(self):
        cam = Camera(name="Test")
        run(db.create_camera(cam))
        for i in range(10):
            run(db.create_alert(Alert(camera_id=cam.id, rule_id="r1", rule_name=f"A{i}", severity="low", timestamp=float(i))))
        page1 = run(db.list_alerts(cam.id, limit=3, offset=0))
        page2 = run(db.list_alerts(cam.id, limit=3, offset=3))
        assert len(page1) == 3
        assert len(page2) == 3
        assert page1[0]["rule_name"] != page2[0]["rule_name"]


class TestUsers:
    def test_create_and_get(self):
        user = User(username="testuser", password_hash="hash123")
        run(db.create_user(user))
        result = run(db.get_user_by_username("testuser"))
        assert result is not None
        assert result.username == "testuser"

    def test_get_nonexistent(self):
        result = run(db.get_user_by_username("nobody"))
        assert result is None

    def test_duplicate_username(self):
        run(db.create_user(User(username="dupe", password_hash="h1")))
        with pytest.raises(Exception):
            run(db.create_user(User(username="dupe", password_hash="h2")))


class TestMemoryEntries:
    def test_create_and_list(self):
        cam = Camera(name="Test")
        run(db.create_camera(cam))
        entry = MemoryEntry(timestamp=time.time(), summary="Person entered", detection_count=1)
        run(db.create_memory_entry(cam.id, entry))
        entries = run(db.list_memory_entries(cam.id))
        assert len(entries) == 1
        assert "entered" in entries[0].summary
