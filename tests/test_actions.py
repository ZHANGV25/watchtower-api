"""Tests for Block 3: Action Engine."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from actions import ActionEngine, DEFAULT_ACTION_CONFIG, TTSResult
from models import Alert


def _make_alert(severity: str = "critical", narration: str = "Person fell") -> Alert:
    return Alert(
        rule_id="test-rule",
        rule_name="Test Alert",
        severity=severity,
        narration=narration,
    )


class TestActionEngine:
    def test_default_config(self):
        engine = ActionEngine()
        assert engine.config == DEFAULT_ACTION_CONFIG
        assert "tts" in engine.config["critical"]
        assert "sound" in engine.config["critical"]
        assert engine.config["low"] == []

    def test_update_config(self):
        engine = ActionEngine()
        new_config = {
            "critical": ["tts", "webhook", "sound"],
            "high": ["sound"],
            "medium": [],
            "low": [],
        }
        engine.update_config(new_config)
        assert engine.config == new_config
        assert "webhook" in engine.config["critical"]

    @pytest.mark.asyncio
    async def test_execute_low_severity_no_actions(self):
        engine = ActionEngine()
        alert = _make_alert(severity="low")

        broadcast_calls: list[tuple[str, dict]] = []

        async def mock_broadcast(event: str, payload: dict):
            broadcast_calls.append((event, payload))

        await engine.execute(alert, mock_broadcast)
        # Low severity has no actions configured
        assert len(broadcast_calls) == 0

    @pytest.mark.asyncio
    async def test_execute_medium_severity_sound(self):
        engine = ActionEngine()
        alert = _make_alert(severity="medium")

        broadcast_calls: list[tuple[str, dict]] = []

        async def mock_broadcast(event: str, payload: dict):
            broadcast_calls.append((event, payload))

        await engine.execute(alert, mock_broadcast)
        # Medium severity triggers sound
        assert len(broadcast_calls) == 1
        assert broadcast_calls[0][0] == "play_sound"
        assert broadcast_calls[0][1]["severity"] == "medium"

    @pytest.mark.asyncio
    async def test_execute_critical_severity_tts_fallback(self):
        """Without ElevenLabs key, should fall back to tts_fallback."""
        engine = ActionEngine()
        engine._elevenlabs_key = ""  # No key
        alert = _make_alert(severity="critical")

        broadcast_calls: list[tuple[str, dict]] = []

        async def mock_broadcast(event: str, payload: dict):
            broadcast_calls.append((event, payload))

        await engine.execute(alert, mock_broadcast)

        event_types = [c[0] for c in broadcast_calls]
        assert "tts_fallback" in event_types
        assert "play_sound" in event_types

    @pytest.mark.asyncio
    async def test_execute_action_error_doesnt_crash(self):
        """If one action fails, others should still execute."""
        engine = ActionEngine()
        engine._elevenlabs_key = "fake-key"

        alert = _make_alert(severity="critical")

        broadcast_calls: list[tuple[str, dict]] = []

        async def mock_broadcast(event: str, payload: dict):
            broadcast_calls.append((event, payload))

        # TTS will fail (fake key), but sound should still work
        await engine.execute(alert, mock_broadcast)
        event_types = [c[0] for c in broadcast_calls]
        assert "play_sound" in event_types


class TestTTSResult:
    def test_defaults(self):
        result = TTSResult(audio_b64="abc123")
        assert result.format == "mp3"
        assert result.audio_b64 == "abc123"
