"""Comprehensive integration tests for all WebSocket handlers.

Tests the full message flow through the actual FastAPI WebSocket endpoint
with mocked camera and LLM calls. No external services needed.

Image inputs: 480x640 solid numpy frames. LLM calls return deterministic mocks.
Anomaly detection uses real feature extraction on synthetic frames.
"""
from __future__ import annotations

import json
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Disable camera before importing main
os.environ["WATCHTOWER_NO_CAMERA"] = "1"

from fastapi.testclient import TestClient

# We need to import main AFTER setting the env var
import main as app_module
from main import app
from models import Alert, BBox, Condition, Detection, Rule, Zone
from scene_analyzer import SceneAnalysis
from narrator import VerificationResult
from reasoner import Insight
from memory import MemoryEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_state():
    """Reset all global state before each test."""
    app_module.zones.clear()
    app_module.rules.clear()
    app_module.alerts.clear()
    app_module.pending_plans.clear()
    app_module.reasoning_enabled = False
    app_module.narration_enabled = False
    app_module.bootstrap_sent = False
    app_module.latest_detections = []
    app_module.latest_frame = None
    app_module.rule_engine._last_fired.clear()
    app_module.rule_engine._duration_tracking.clear()
    app_module.anomaly_detector.stop()
    yield


@pytest.fixture
def client():
    return TestClient(app)


def _make_frame(r: int = 0, g: int = 0, b: int = 0) -> np.ndarray:
    """Create a solid-color 480x640 frame."""
    frame = np.full((480, 640, 3), [b, g, r], dtype=np.uint8)
    return frame


def _make_detection(cls: str = "person", conf: float = 0.9) -> Detection:
    return Detection(
        class_name=cls,
        confidence=conf,
        bbox=BBox(x=30, y=20, width=15, height=40),
    )


def _make_rule(name: str = "Test Rule", severity: str = "medium") -> Rule:
    return Rule(
        name=name,
        natural_language=f"Alert for {name}",
        conditions=[Condition(type="object_present", params={"class": "person"})],
        severity=severity,
    )


def _make_zone(name: str = "Test Zone") -> Zone:
    return Zone(name=name, x=10, y=20, width=30, height=40)


# ---------------------------------------------------------------------------
# Init / Connection
# ---------------------------------------------------------------------------

class TestConnection:
    def test_connect_receives_init(self, client):
        """WebSocket connection should immediately receive init with state."""
        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert data["type"] == "init"
            payload = data["payload"]
            assert "zones" in payload
            assert "rules" in payload
            assert "alerts" in payload
            assert "reasoning_enabled" in payload
            assert "narration_enabled" in payload
            assert "anomaly_phase" in payload
            assert "action_config" in payload

    def test_init_reflects_current_state(self, client):
        """Init payload should reflect zones/rules added before connection."""
        app_module.zones.append(_make_zone("Kitchen"))
        app_module.rules.append(_make_rule("Watch kitchen"))
        app_module.reasoning_enabled = True

        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert len(data["payload"]["zones"]) == 1
            assert data["payload"]["zones"][0]["name"] == "Kitchen"
            assert len(data["payload"]["rules"]) == 1
            assert data["payload"]["reasoning_enabled"] is True

    def test_init_anomaly_phase(self, client):
        """Init payload should report anomaly phase."""
        app_module.anomaly_detector.start_learning()
        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert data["payload"]["anomaly_phase"] == "learning"

    def test_init_action_config(self, client):
        """Init payload should include action configuration."""
        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            config = data["payload"]["action_config"]
            assert "critical" in config
            assert "tts" in config["critical"]


# ---------------------------------------------------------------------------
# Rule Handlers
# ---------------------------------------------------------------------------

class TestAddRule:
    def test_add_rule_success(self, client):
        """add_rule with valid text should return rule_added."""
        mock_rule = _make_rule("Person Alert")

        with patch.object(app_module.rule_parser, "parse", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = (mock_rule, [])

            with client.websocket_connect("/ws") as ws:
                ws.receive_json()  # init
                ws.send_json({"type": "add_rule", "payload": {"text": "alert if person", "severity": "high"}})
                data = ws.receive_json()
                assert data["type"] == "rule_added"
                assert data["payload"]["name"] == "Person Alert"

    def test_add_rule_empty_text(self, client):
        """add_rule with empty text should not produce a response."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # init
            ws.send_json({"type": "add_rule", "payload": {"text": ""}})
            # Should not receive rule_added — send another message to verify
            ws.send_json({"type": "clear_alerts", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "alerts_cleared"

    def test_add_rule_with_missing_zones(self, client):
        """add_rule referencing nonexistent zones should include _missing_zones."""
        mock_rule = _make_rule("Door Watch")

        with patch.object(app_module.rule_parser, "parse", new_callable=AsyncMock) as mock_parse:
            mock_parse.return_value = (mock_rule, ["Front Door"])

            with client.websocket_connect("/ws") as ws:
                ws.receive_json()
                ws.send_json({"type": "add_rule", "payload": {"text": "watch the front door"}})
                data = ws.receive_json()
                assert data["type"] == "rule_added"
                assert "_missing_zones" in data["payload"]
                assert "Front Door" in data["payload"]["_missing_zones"]


class TestToggleRule:
    def test_toggle_rule(self, client):
        """toggle_rule should flip enabled and broadcast rule_updated."""
        rule = _make_rule()
        app_module.rules.append(rule)

        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # init
            ws.send_json({"type": "toggle_rule", "payload": {"id": rule.id}})
            data = ws.receive_json()
            assert data["type"] == "rule_updated"
            assert data["payload"]["enabled"] is False

    def test_toggle_rule_back_on(self, client):
        """Toggling twice should re-enable."""
        rule = _make_rule()
        rule.enabled = False
        app_module.rules.append(rule)

        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "toggle_rule", "payload": {"id": rule.id}})
            data = ws.receive_json()
            assert data["payload"]["enabled"] is True

    def test_toggle_nonexistent_rule(self, client):
        """Toggling a nonexistent rule should not crash."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "toggle_rule", "payload": {"id": "nonexistent"}})
            # Should not crash — send ping to verify connection alive
            ws.send_json({"type": "clear_alerts", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "alerts_cleared"


class TestDeleteRule:
    def test_delete_rule(self, client):
        """delete_rule should remove rule and broadcast rule_deleted."""
        rule = _make_rule()
        app_module.rules.append(rule)

        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "delete_rule", "payload": {"id": rule.id}})
            data = ws.receive_json()
            assert data["type"] == "rule_deleted"
            assert data["payload"]["id"] == rule.id
            assert len(app_module.rules) == 0


class TestUpdateRule:
    def test_update_rule_name(self, client):
        """update_rule should modify fields and broadcast."""
        rule = _make_rule("Original")
        app_module.rules.append(rule)

        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "update_rule", "payload": {"id": rule.id, "name": "Updated"}})
            data = ws.receive_json()
            assert data["type"] == "rule_updated"
            assert data["payload"]["name"] == "Updated"


class TestClearRules:
    def test_clear_rules(self, client):
        """clear_rules should empty rules list and broadcast."""
        app_module.rules.extend([_make_rule("A"), _make_rule("B")])
        assert len(app_module.rules) == 2

        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "clear_rules", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "rules_cleared"
            assert len(app_module.rules) == 0


# ---------------------------------------------------------------------------
# Zone Handlers
# ---------------------------------------------------------------------------

class TestUpdateZones:
    def test_update_zones(self, client):
        """update_zones should replace zones and broadcast."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            new_zones = [
                {"id": "z1", "name": "Door", "x": 10, "y": 20, "width": 30, "height": 40, "color": "#22d3ee"},
                {"id": "z2", "name": "Window", "x": 50, "y": 10, "width": 20, "height": 30, "color": "#a78bfa"},
            ]
            ws.send_json({"type": "update_zones", "payload": {"zones": new_zones}})
            data = ws.receive_json()
            assert data["type"] == "zones_updated"
            assert len(data["payload"]["zones"]) == 2
            assert data["payload"]["zones"][0]["name"] == "Door"

    def test_update_zones_empty(self, client):
        """Sending empty zones should clear all zones."""
        app_module.zones.append(_make_zone())
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "update_zones", "payload": {"zones": []}})
            data = ws.receive_json()
            assert data["type"] == "zones_updated"
            assert len(data["payload"]["zones"]) == 0


class TestAutoZones:
    def test_auto_zones_no_camera(self, client):
        """auto_zones without camera should not crash."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "auto_zones", "payload": {}})
            # No response expected (camera is None), verify connection alive
            ws.send_json({"type": "clear_alerts", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "alerts_cleared"


# ---------------------------------------------------------------------------
# Alert Handlers
# ---------------------------------------------------------------------------

class TestClearAlerts:
    def test_clear_alerts(self, client):
        """clear_alerts should empty alerts and broadcast."""
        app_module.alerts.append(Alert(
            rule_id="r1", rule_name="Test", severity="low",
        ))

        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "clear_alerts", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "alerts_cleared"
            assert len(app_module.alerts) == 0


# ---------------------------------------------------------------------------
# Replay Handlers
# ---------------------------------------------------------------------------

class TestReplay:
    def test_get_replay_timestamps(self, client):
        """get_replay_timestamps should return buffer info."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "get_replay_timestamps", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "replay_timestamps"
            assert "start" in data["payload"]
            assert "end" in data["payload"]
            assert "count" in data["payload"]

    def test_get_frame_at(self, client):
        """get_frame_at should return a replay_frame message."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "get_frame_at", "payload": {"timestamp": time.time()}})
            data = ws.receive_json()
            assert data["type"] == "replay_frame"
            assert "frame" in data["payload"]
            assert "timestamp" in data["payload"]

    def test_get_replay_range(self, client):
        """get_replay should return frames for a time range."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "get_replay", "payload": {"timestamp": time.time(), "duration": 5}})
            data = ws.receive_json()
            assert data["type"] == "replay"
            assert "frames" in data["payload"]


# ---------------------------------------------------------------------------
# Block 1: Bootstrap
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_approve_bootstrap_creates_zones_and_rules(self, client):
        """approve_bootstrap should create zones and rules from payload."""
        zones = [
            {"name": "Door", "x": 10, "y": 20, "width": 30, "height": 40},
            {"name": "Stairs", "x": 50, "y": 10, "width": 20, "height": 60},
        ]
        rules = [
            {
                "name": "Fall detection",
                "natural_language": "Alert if someone falls",
                "conditions": [{"type": "person_pose", "params": {"pose": "lying"}}],
                "severity": "critical",
            },
        ]

        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # init
            ws.send_json({"type": "approve_bootstrap", "payload": {"zones": zones, "rules": rules}})

            # Should receive zones_updated
            data = ws.receive_json()
            assert data["type"] == "zones_updated"
            assert len(data["payload"]["zones"]) == 2

            # Should receive rule_added
            data = ws.receive_json()
            assert data["type"] == "rule_added"
            assert data["payload"]["name"] == "Fall detection"
            assert data["payload"]["severity"] == "critical"

        assert len(app_module.zones) == 2
        assert len(app_module.rules) == 1

    def test_approve_bootstrap_empty(self, client):
        """approve_bootstrap with empty lists should create nothing."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "approve_bootstrap", "payload": {"zones": [], "rules": []}})
            data = ws.receive_json()
            assert data["type"] == "zones_updated"
            assert len(data["payload"]["zones"]) == 0

    def test_dismiss_bootstrap(self, client):
        """dismiss_bootstrap should not crash and not change state."""
        app_module.zones.append(_make_zone())
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "dismiss_bootstrap", "payload": {}})
            # Verify state unchanged
            ws.send_json({"type": "clear_alerts", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "alerts_cleared"
            assert len(app_module.zones) == 1  # unchanged


# ---------------------------------------------------------------------------
# Block 2: Reasoning
# ---------------------------------------------------------------------------

class TestReasoning:
    def test_toggle_reasoning_on(self, client):
        """toggle_reasoning should enable and broadcast."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "toggle_reasoning", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "reasoning_toggled"
            assert data["payload"]["enabled"] is True
            assert app_module.reasoning_enabled is True

    def test_toggle_reasoning_off(self, client):
        """Toggling again should disable."""
        app_module.reasoning_enabled = True
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "toggle_reasoning", "payload": {}})
            data = ws.receive_json()
            assert data["payload"]["enabled"] is False

    def test_reasoning_reflected_in_init(self, client):
        """Reasoning state should be in init payload."""
        app_module.reasoning_enabled = True
        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert data["payload"]["reasoning_enabled"] is True


# ---------------------------------------------------------------------------
# Block 3: Actions
# ---------------------------------------------------------------------------

class TestActions:
    def test_update_actions(self, client):
        """update_actions should change config and broadcast."""
        new_config = {
            "critical": ["tts", "webhook", "sound"],
            "high": ["sound"],
            "medium": [],
            "low": [],
        }
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "update_actions", "payload": {"config": new_config}})
            data = ws.receive_json()
            assert data["type"] == "actions_updated"
            assert "webhook" in data["payload"]["config"]["critical"]
            assert data["payload"]["config"]["medium"] == []

    def test_action_config_persists_across_connections(self, client):
        """Updated config should be in init for new connections."""
        app_module.action_engine.update_config({
            "critical": ["webhook"],
            "high": [],
            "medium": [],
            "low": [],
        })
        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert data["payload"]["action_config"]["critical"] == ["webhook"]


# ---------------------------------------------------------------------------
# Block 4: Investigation
# ---------------------------------------------------------------------------

class TestInvestigation:
    def test_ask_returns_answer(self, client):
        """ask should invoke memory investigation and return answer."""
        with patch.object(
            app_module.scene_memory, "investigate", new_callable=AsyncMock
        ) as mock_investigate:
            mock_investigate.return_value = "A person entered the room at 14:32 and sat down."

            with client.websocket_connect("/ws") as ws:
                ws.receive_json()
                ws.send_json({"type": "ask", "payload": {"question": "What happened?"}})
                data = ws.receive_json()
                assert data["type"] == "ask_response"
                assert "person entered" in data["payload"]["answer"]
                assert data["payload"]["question"] == "What happened?"

    def test_ask_empty_question(self, client):
        """Empty question should not produce a response."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "ask", "payload": {"question": ""}})
            ws.send_json({"type": "clear_alerts", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "alerts_cleared"

    def test_ask_includes_relevant_frames(self, client):
        """ask response should include frames when available."""
        app_module.latest_frame = _make_frame(128, 128, 128)

        with patch.object(
            app_module.scene_memory, "investigate", new_callable=AsyncMock
        ) as mock_investigate:
            mock_investigate.return_value = "Nothing notable happened."

            with client.websocket_connect("/ws") as ws:
                ws.receive_json()
                ws.send_json({"type": "ask", "payload": {"question": "Anything happen?"}})
                data = ws.receive_json()
                assert data["type"] == "ask_response"
                assert len(data["payload"]["relevant_frames"]) > 0
                assert data["payload"]["relevant_frames"][0]["frame"] != ""


# ---------------------------------------------------------------------------
# Block 5: Narration
# ---------------------------------------------------------------------------

class TestNarration:
    def test_toggle_narration_on(self, client):
        """toggle_narration should enable and broadcast."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "toggle_narration", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "narration_toggled"
            assert data["payload"]["enabled"] is True

    def test_toggle_narration_off(self, client):
        """Toggling again should disable."""
        app_module.narration_enabled = True
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "toggle_narration", "payload": {}})
            data = ws.receive_json()
            assert data["payload"]["enabled"] is False

    def test_narration_reflected_in_init(self, client):
        """Narration state should be in init payload."""
        app_module.narration_enabled = True
        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert data["payload"]["narration_enabled"] is True


# ---------------------------------------------------------------------------
# Block 6: Anomaly
# ---------------------------------------------------------------------------

class TestAnomaly:
    def test_toggle_anomaly_starts_learning(self, client):
        """toggle_anomaly from off should start learning phase."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "toggle_anomaly", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "anomaly_status"
            assert data["payload"]["phase"] == "learning"
            assert data["payload"]["time_remaining"] > 0

    def test_toggle_anomaly_stops(self, client):
        """toggle_anomaly from learning/detecting should stop."""
        app_module.anomaly_detector.start_learning()
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "toggle_anomaly", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "anomaly_status"
            assert data["payload"]["phase"] == "off"

    def test_set_anomaly_threshold(self, client):
        """set_anomaly_threshold should update threshold."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "set_anomaly_threshold", "payload": {"threshold": 0.5}})
            data = ws.receive_json()
            assert data["type"] == "anomaly_status"
            assert app_module.anomaly_detector.threshold == 0.5

    def test_set_anomaly_threshold_clamped(self, client):
        """Threshold should be clamped to valid range."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "set_anomaly_threshold", "payload": {"threshold": 2.0}})
            ws.receive_json()
            assert app_module.anomaly_detector.threshold == 0.95

    def test_anomaly_phase_in_init(self, client):
        """Anomaly phase should be in init payload."""
        app_module.anomaly_detector.start_learning()
        with client.websocket_connect("/ws") as ws:
            data = ws.receive_json()
            assert data["payload"]["anomaly_phase"] == "learning"


# ---------------------------------------------------------------------------
# Plan Handlers (existing feature, preserved)
# ---------------------------------------------------------------------------

class TestPlanHandlers:
    def test_generate_plan_no_camera(self, client):
        """generate_plan without camera should still attempt LLM classification."""
        with patch.object(
            app_module.plan_generator, "classify_and_generate", new_callable=AsyncMock
        ) as mock_gen:
            mock_rule = _make_rule("Generated Rule")
            mock_gen.return_value = {"type": "rule", "rule": mock_rule, "missing_zones": []}

            with client.websocket_connect("/ws") as ws:
                ws.receive_json()
                ws.send_json({"type": "generate_plan", "payload": {"text": "watch for intruders"}})
                data = ws.receive_json()
                assert data["type"] == "rule_added"

    def test_generate_plan_empty_text(self, client):
        """generate_plan with empty text should not crash."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "generate_plan", "payload": {"text": ""}})
            ws.send_json({"type": "clear_alerts", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "alerts_cleared"


# ---------------------------------------------------------------------------
# Unknown Message Type
# ---------------------------------------------------------------------------

class TestUnknownMessage:
    def test_unknown_type_no_crash(self, client):
        """Sending unknown message type should not crash the connection."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "totally_fake_message", "payload": {}})
            # Connection should still be alive
            ws.send_json({"type": "clear_alerts", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "alerts_cleared"


# ---------------------------------------------------------------------------
# Cross-Feature: Multiple Toggles
# ---------------------------------------------------------------------------

class TestCrossFeature:
    def test_multiple_toggles_simultaneously(self, client):
        """Enabling multiple features at once should work."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()

            ws.send_json({"type": "toggle_reasoning", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "reasoning_toggled"
            assert data["payload"]["enabled"] is True

            ws.send_json({"type": "toggle_narration", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "narration_toggled"
            assert data["payload"]["enabled"] is True

            ws.send_json({"type": "toggle_anomaly", "payload": {}})
            data = ws.receive_json()
            assert data["type"] == "anomaly_status"
            assert data["payload"]["phase"] == "learning"

            # All three enabled
            assert app_module.reasoning_enabled is True
            assert app_module.narration_enabled is True
            assert app_module.anomaly_detector.phase.value == "learning"

    def test_full_flow_bootstrap_to_alert(self, client):
        """End-to-end: bootstrap → zones + rules created → verify state."""
        zones = [{"name": "Door", "x": 0, "y": 0, "width": 50, "height": 100}]
        rules = [{
            "name": "Person at door",
            "natural_language": "Alert if person near door",
            "conditions": [
                {"type": "object_in_zone", "params": {"class": "person", "zone": "Door"}},
            ],
            "severity": "high",
        }]

        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "approve_bootstrap", "payload": {"zones": zones, "rules": rules}})

            # zones_updated
            data = ws.receive_json()
            assert data["type"] == "zones_updated"

            # rule_added
            data = ws.receive_json()
            assert data["type"] == "rule_added"

        # Verify final state
        assert len(app_module.zones) == 1
        assert app_module.zones[0].name == "Door"
        assert len(app_module.rules) == 1
        assert app_module.rules[0].conditions[0].type == "object_in_zone"

    def test_state_isolation_between_connections(self, client):
        """Changes from one connection should be visible in new connections."""
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "toggle_reasoning", "payload": {}})
            ws.receive_json()

        # New connection should see reasoning enabled
        with client.websocket_connect("/ws") as ws2:
            data = ws2.receive_json()
            assert data["payload"]["reasoning_enabled"] is True
