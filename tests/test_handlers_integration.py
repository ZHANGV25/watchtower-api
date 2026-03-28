"""Integration tests for all WebSocket handlers.

Tests the full message flow through the FastAPI WebSocket endpoint
with the new multi-camera + database architecture.
"""
from __future__ import annotations

import json
import os
import sys
import time
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ["WATCHTOWER_NO_CAMERA"] = "1"
os.environ["WATCHTOWER_DB"] = "/tmp/watchtower_handler_test.db"

from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset DB and camera manager before each test."""
    try:
        os.remove("/tmp/watchtower_handler_test.db")
    except FileNotFoundError:
        pass

    import database as db
    import main as app_module
    db._db = None
    app_module.camera_mgr._sessions.clear()
    app_module.frontend_clients.clear()
    app_module.pending_plans.clear()
    yield


@pytest.fixture
def client():
    from main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# Connection + Init
# ---------------------------------------------------------------------------

class TestConnection:
    def test_connect_receives_init(self, client):
        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert data["type"] == "init"
            p = data["payload"]
            assert "cameras" in p
            assert "camera_id" in p
            assert "zones" in p
            assert "rules" in p
            assert "alerts" in p
            assert "reasoning_enabled" in p
            assert "narration_enabled" in p
            assert "anomaly_phase" in p

    def test_init_has_default_camera(self, client):
        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert len(data["payload"]["cameras"]) >= 1


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

class TestRules:
    def test_add_rule(self, client):
        from models import Rule
        mock_rule = Rule(name="Test", natural_language="test", conditions=[], severity="medium")

        with patch("main.rule_parser.parse", new_callable=AsyncMock) as mock:
            mock.return_value = (mock_rule, [])
            with client.websocket_connect("/ws") as ws:
                ws.receive_json()  # init
                ws.send_json({"type": "add_rule", "payload": {"text": "test rule"}})
                data = ws.receive_json()
                assert data["type"] == "rule_added"
                assert data["payload"]["name"] == "Test"

    def test_toggle_rule(self, client):
        from models import Condition, Rule
        import main as m
        with client.websocket_connect("/ws") as ws:
            init = ws.receive_json()
            cam_id = init["payload"]["camera_id"]
            session = m.camera_mgr.get_session(cam_id)
            rule = Rule(camera_id=cam_id, name="R1", natural_language="t", conditions=[])
            session.rules.append(rule)

            ws.send_json({"type": "toggle_rule", "payload": {"id": rule.id}})
            data = ws.receive_json()
            assert data["type"] == "rule_updated"
            assert data["payload"]["enabled"] is False

    def test_delete_rule(self, client):
        from models import Rule
        import main as m
        with client.websocket_connect("/ws") as ws:
            init = ws.receive_json()
            cam_id = init["payload"]["camera_id"]
            session = m.camera_mgr.get_session(cam_id)
            rule = Rule(camera_id=cam_id, name="R1", natural_language="t", conditions=[])
            session.rules.append(rule)

            ws.send_json({"type": "delete_rule", "payload": {"id": rule.id}})
            data = ws.receive_json()
            assert data["type"] == "rule_deleted"
            assert len(session.rules) == 0

    def test_clear_rules(self, client):
        from models import Rule
        import main as m
        with client.websocket_connect("/ws") as ws:
            init = ws.receive_json()
            cam_id = init["payload"]["camera_id"]
            session = m.camera_mgr.get_session(cam_id)
            session.rules.append(Rule(camera_id=cam_id, name="R1", natural_language="t", conditions=[]))

            ws.send_json({"type": "clear_rules", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "rules_cleared"
            assert len(session.rules) == 0


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------

class TestZones:
    def test_update_zones(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            zones = [{"id": "z1", "name": "Door", "x": 10, "y": 20, "width": 30, "height": 40, "color": "#22d3ee"}]
            ws.send_json({"type": "update_zones", "payload": {"zones": zones}})
            data = ws.receive_json()
            assert data["type"] == "zones_updated"
            assert len(data["payload"]["zones"]) == 1


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

class TestAlerts:
    def test_clear_alerts(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "clear_alerts", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "alerts_cleared"


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

class TestReplay:
    def test_get_replay_timestamps(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "get_replay_timestamps", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "replay_timestamps"

    def test_get_frame_at(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "get_frame_at", "payload": {"timestamp": time.time()}})
            data = ws.receive_json()
            assert data["type"] == "replay_frame"


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_approve_bootstrap(self, client):
        zones = [{"name": "Door", "x": 10, "y": 20, "width": 30, "height": 40}]
        rules = [{"name": "Fall", "natural_language": "Detect falls",
                  "conditions": [{"type": "person_pose", "params": {"pose": "lying"}}], "severity": "critical"}]

        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "approve_bootstrap", "payload": {"zones": zones, "rules": rules}})
            d1 = ws.receive_json()
            assert d1["type"] == "zones_updated"
            d2 = ws.receive_json()
            assert d2["type"] == "rule_added"

    def test_dismiss_bootstrap(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "dismiss_bootstrap", "payload": {}})
            # Should not crash
            ws.send_json({"type": "clear_alerts", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "alerts_cleared"


# ---------------------------------------------------------------------------
# Feature Toggles
# ---------------------------------------------------------------------------

class TestToggles:
    def test_toggle_reasoning(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "toggle_reasoning", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "reasoning_toggled"
            assert data["payload"]["enabled"] is True

    def test_toggle_narration(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "toggle_narration", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "narration_toggled"
            assert data["payload"]["enabled"] is True

    def test_toggle_anomaly(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "toggle_anomaly", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "anomaly_status"
            assert data["payload"]["phase"] == "learning"

    def test_set_anomaly_threshold(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "set_anomaly_threshold", "payload": {"threshold": 0.5}})
            data = ws.receive_json()
            assert data["type"] == "anomaly_status"


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class TestActions:
    def test_update_actions(self, client):
        config = {"critical": ["tts", "webhook"], "high": [], "medium": [], "low": []}
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "update_actions", "payload": {"config": config}})
            data = ws.receive_json()
            assert data["type"] == "actions_updated"
            assert "webhook" in data["payload"]["config"]["critical"]


# ---------------------------------------------------------------------------
# Investigation
# ---------------------------------------------------------------------------

class TestInvestigation:
    def test_ask(self, client):
        import main as m
        with patch.object(m.camera_mgr.get_or_create_session("default").scene_memory, "investigate", new_callable=AsyncMock) as mock:
            mock.return_value = "A person entered at 14:32."
            with client.websocket_connect("/ws") as ws:
                init = ws.receive_json()
                cam_id = init["payload"]["camera_id"]
                session = m.camera_mgr.get_session(cam_id)
                with patch.object(session.scene_memory, "investigate", new_callable=AsyncMock) as mock2:
                    mock2.return_value = "A person entered at 14:32."
                    ws.send_json({"type": "ask", "payload": {"question": "What happened?"}})
                    data = ws.receive_json()
                    assert data["type"] == "ask_response"
                    assert "14:32" in data["payload"]["answer"]


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_all(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "reset_all", "payload": {}})
            # Should receive multiple clear messages
            messages = []
            for _ in range(6):
                messages.append(ws.receive_json())
            types = {m["type"] for m in messages}
            assert "zones_updated" in types
            assert "rules_cleared" in types
            assert "alerts_cleared" in types


# ---------------------------------------------------------------------------
# Subscribe (camera switching)
# ---------------------------------------------------------------------------

class TestSubscribe:
    def test_subscribe_to_existing_camera(self, client):
        """Subscribe to the default camera (already exists from init)."""
        import main as m

        with client.websocket_connect("/ws") as ws:
            init = ws.receive_json()
            cam_id = init["payload"]["camera_id"]

            # Create a second session in memory (no DB needed)
            session2 = m.camera_mgr.get_or_create_session("cam2", "Second Camera")

            # Subscribe to it
            ws.send_json({"type": "subscribe", "payload": {"camera_id": "cam2"}})
            data = ws.receive_json()
            assert data["type"] == "init"
            assert data["payload"]["camera_id"] == "cam2"


# ---------------------------------------------------------------------------
# Unknown Message
# ---------------------------------------------------------------------------

class TestUnknown:
    def test_unknown_no_crash(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "totally_fake", "payload": {}})
            ws.send_json({"type": "clear_alerts", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "alerts_cleared"
