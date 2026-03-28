"""Block 3: Agentic actions.

When alerts fire, WatchTower takes action: TTS via ElevenLabs,
webhooks, browser sound notifications.
"""
from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import httpx

from models import Alert

log = logging.getLogger("watchtower.actions")

# Default action configuration per severity level
DEFAULT_ACTION_CONFIG: dict[str, list[str]] = {
    "critical": ["tts", "sound"],
    "high": ["tts", "sound"],
    "medium": ["sound"],
    "low": [],
}


@dataclass
class TTSResult:
    audio_b64: str
    format: str = "mp3"


class ActionEngine:
    def __init__(self) -> None:
        self._config: dict[str, list[str]] = dict(DEFAULT_ACTION_CONFIG)
        self._elevenlabs_key = os.getenv("ELEVENLABS_API_KEY", "")
        self._elevenlabs_voice = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
        self._webhook_url = os.getenv("WATCHTOWER_WEBHOOK_URL", "")

    @property
    def config(self) -> dict[str, list[str]]:
        return self._config

    def update_config(self, config: dict[str, list[str]]) -> None:
        self._config = config

    async def execute(
        self,
        alert: Alert,
        broadcast_fn: Callable[[str, dict[str, Any]], Coroutine],
    ) -> None:
        """Execute configured actions for an alert's severity level."""
        actions = self._config.get(alert.severity, [])
        if not actions:
            return

        for action in actions:
            try:
                if action == "tts":
                    await self._do_tts(alert, broadcast_fn)
                elif action == "webhook":
                    await self._do_webhook(alert)
                elif action == "sound":
                    await self._do_sound(alert, broadcast_fn)
            except Exception as e:
                log.error("Action '%s' failed for alert '%s': %s", action, alert.rule_name, e)

    async def _do_tts(
        self,
        alert: Alert,
        broadcast_fn: Callable[[str, dict[str, Any]], Coroutine],
    ) -> None:
        """Generate TTS audio via ElevenLabs and broadcast to clients."""
        text = alert.narration or f"Alert: {alert.rule_name}"
        if not text:
            return

        if self._elevenlabs_key:
            result = await self._elevenlabs_tts(text)
            if result:
                await broadcast_fn("tts_audio", {
                    "audio_b64": result.audio_b64,
                    "format": result.format,
                    "alert_id": alert.id,
                })
                return

        # Fallback: send text for browser Web Speech API
        await broadcast_fn("tts_fallback", {
            "text": text,
            "alert_id": alert.id,
        })

    async def _elevenlabs_tts(self, text: str) -> TTSResult | None:
        """Call ElevenLabs API for text-to-speech."""
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._elevenlabs_voice}"
        headers = {
            "xi-api-key": self._elevenlabs_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": "eleven_flash_v2_5",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                audio_b64 = base64.b64encode(resp.content).decode("ascii")
                return TTSResult(audio_b64=audio_b64, format="mp3")
        except Exception as e:
            log.error("ElevenLabs TTS failed: %s", e)
            return None

    async def _do_webhook(self, alert: Alert) -> None:
        """POST alert data to configured webhook URL."""
        if not self._webhook_url:
            return

        payload = {
            "id": alert.id,
            "rule_name": alert.rule_name,
            "severity": alert.severity,
            "narration": alert.narration,
            "timestamp": alert.timestamp,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(self._webhook_url, json=payload)
        except Exception as e:
            log.error("Webhook failed: %s", e)

    async def _do_sound(
        self,
        alert: Alert,
        broadcast_fn: Callable[[str, dict[str, Any]], Coroutine],
    ) -> None:
        """Tell frontend to play a notification sound."""
        await broadcast_fn("play_sound", {"severity": alert.severity})
