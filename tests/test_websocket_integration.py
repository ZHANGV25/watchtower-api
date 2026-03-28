"""Integration tests for the WebSocket API.

Tests the full flow: connect, send add_rule, receive rule_added response.
Requires the backend to be running on localhost:8000.
"""
from __future__ import annotations

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pytest_asyncio

try:
    import websockets
except ImportError:
    websockets = None

WS_URL = "ws://localhost:8000/ws"


@pytest_asyncio.fixture
async def ws():
    """Connect a WebSocket client to the running backend."""
    if websockets is None:
        pytest.skip("websockets not installed")
    async with websockets.connect(WS_URL) as conn:
        # Consume messages until we get init (alerts may arrive first from prior tests)
        deadline = asyncio.get_event_loop().time() + 5
        while asyncio.get_event_loop().time() < deadline:
            raw = await asyncio.wait_for(conn.recv(), timeout=5)
            msg = json.loads(raw)
            if msg["type"] == "init":
                break
        yield conn


def _make_msg(msg_type: str, payload: dict) -> str:
    return json.dumps({"type": msg_type, "payload": payload})


@pytest.mark.asyncio
class TestWebSocketConnection:
    async def test_connect_and_receive_init(self):
        """Connecting should immediately receive an init message with zones, rules, alerts."""
        async with websockets.connect(WS_URL) as conn:
            raw = await asyncio.wait_for(conn.recv(), timeout=5)
            msg = json.loads(raw)
            assert msg["type"] == "init"
            assert "zones" in msg["payload"]
            assert "rules" in msg["payload"]
            assert "alerts" in msg["payload"]

    async def test_receives_frames(self):
        """After connect, should receive frame messages if camera is active."""
        async with websockets.connect(WS_URL) as conn:
            # Consume init
            await asyncio.wait_for(conn.recv(), timeout=5)
            # Wait for a frame (may not arrive if camera is disabled)
            try:
                raw = await asyncio.wait_for(conn.recv(), timeout=5)
                msg = json.loads(raw)
                if msg["type"] == "frame":
                    assert "frame" in msg["payload"]
                    assert "detections" in msg["payload"]
                    assert "fps" in msg["payload"]
                # else: got a different message type (e.g. alert), that's fine
            except asyncio.TimeoutError:
                # No frames received -- camera is likely disabled, skip
                pytest.skip("No frames received (camera may be disabled)")


@pytest.mark.asyncio
class TestAddRule:
    async def test_add_rule_returns_rule_added(self, ws):
        """Sending add_rule should eventually receive rule_added broadcast."""
        await ws.send(_make_msg("add_rule", {"text": "Alert if a person is detected"}))

        # Collect messages until we get rule_added (skip frame messages)
        deadline = asyncio.get_event_loop().time() + 30  # 30s timeout for LLM
        while asyncio.get_event_loop().time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                msg = json.loads(raw)
                if msg["type"] == "rule_added":
                    rule = msg["payload"]
                    assert "id" in rule
                    assert "name" in rule
                    assert "conditions" in rule
                    assert rule["natural_language"] == "Alert if a person is detected"
                    assert rule["enabled"] is True
                    return  # Success
            except asyncio.TimeoutError:
                continue

        pytest.fail("Never received rule_added message within 30 seconds")

    async def test_add_rule_empty_text_no_response(self, ws):
        """Empty text should not produce a rule_added."""
        await ws.send(_make_msg("add_rule", {"text": ""}))

        # Collect for 3 seconds, should only get frames
        end = asyncio.get_event_loop().time() + 3
        while asyncio.get_event_loop().time() < end:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1)
                msg = json.loads(raw)
                assert msg["type"] != "rule_added", "Should not add rule for empty text"
            except asyncio.TimeoutError:
                continue


@pytest.mark.asyncio
class TestToggleDeleteRule:
    async def test_toggle_and_delete(self, ws):
        """Add a rule, toggle it, then delete it."""
        # Add rule
        await ws.send(_make_msg("add_rule", {"text": "Alert if a cat is detected"}))

        rule_id = None
        deadline = asyncio.get_event_loop().time() + 30
        while asyncio.get_event_loop().time() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=5)
            msg = json.loads(raw)
            if msg["type"] == "rule_added":
                rule_id = msg["payload"]["id"]
                break

        assert rule_id, "Rule was not added"

        # Toggle it off
        await ws.send(_make_msg("toggle_rule", {"id": rule_id}))
        deadline = asyncio.get_event_loop().time() + 5
        while asyncio.get_event_loop().time() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=2)
            msg = json.loads(raw)
            if msg["type"] == "rule_updated":
                assert msg["payload"]["enabled"] is False
                break

        # Delete it
        await ws.send(_make_msg("delete_rule", {"id": rule_id}))
        deadline = asyncio.get_event_loop().time() + 5
        while asyncio.get_event_loop().time() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=2)
            msg = json.loads(raw)
            if msg["type"] == "rule_deleted":
                assert msg["payload"]["id"] == rule_id
                return

        pytest.fail("Did not receive rule_deleted")


async def _wait_for_msg(ws, msg_type: str, timeout: float = 10) -> dict | None:
    """Consume messages until we find one of the given type, or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=2)
            msg = json.loads(raw)
            if msg["type"] == msg_type:
                return msg
        except asyncio.TimeoutError:
            continue
    return None


@pytest.mark.asyncio
class TestClearAlerts:
    async def test_clear_alerts(self, ws):
        """Sending clear_alerts should receive alerts_cleared."""
        await ws.send(_make_msg("clear_alerts", {}))
        msg = await _wait_for_msg(ws, "alerts_cleared", timeout=5)
        assert msg is not None, "Did not receive alerts_cleared"
        assert msg["type"] == "alerts_cleared"


@pytest.mark.asyncio
class TestClearRules:
    async def test_clear_rules(self, ws):
        """Sending clear_rules should receive rules_cleared."""
        # First add a rule so there's something to clear
        await ws.send(_make_msg("add_rule", {"text": "Alert if a dog is detected"}))
        added = await _wait_for_msg(ws, "rule_added", timeout=30)
        assert added is not None, "Rule was not added"

        # Now clear all rules
        await ws.send(_make_msg("clear_rules", {}))
        msg = await _wait_for_msg(ws, "rules_cleared", timeout=5)
        assert msg is not None, "Did not receive rules_cleared"

        # Verify: reconnect and check init has empty rules
        async with websockets.connect(WS_URL) as conn2:
            raw = await asyncio.wait_for(conn2.recv(), timeout=5)
            init = json.loads(raw)
            assert init["type"] == "init"
            assert len(init["payload"]["rules"]) == 0, "Rules should be empty after clear"


@pytest.mark.asyncio
class TestGetFrameAt:
    async def test_get_frame_at_returns_replay_frame(self, ws):
        """Requesting a frame at a timestamp should return a replay_frame message."""
        import time
        await ws.send(_make_msg("get_frame_at", {"timestamp": time.time()}))
        msg = await _wait_for_msg(ws, "replay_frame", timeout=5)
        assert msg is not None, "Did not receive replay_frame"
        assert "frame" in msg["payload"]
        assert "timestamp" in msg["payload"]
