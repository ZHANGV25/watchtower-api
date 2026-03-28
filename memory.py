"""Block 4: Scene memory and conversational investigation.

Maintains a rolling text log of scene summaries (one per ~30s) that
enables natural language questions about what happened in the past.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import anthropic
import cv2
import numpy as np

from models import Alert, Detection, MemoryEntry

log = logging.getLogger("watchtower.memory")

_BEDROCK_MODEL_FAST = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
_BEDROCK_MODEL_SMART = "us.anthropic.claude-sonnet-4-6"

_SUMMARIZE_PROMPT = """\
You are a scene logger for a camera monitoring system. Your job is to record what is happening RIGHT NOW in one concise sentence, focusing on what has CHANGED or what is noteworthy.

Focus on: who is present, what they are doing, any new objects or people, anyone leaving or arriving, changes in activity.

If the scene is static and unchanged, respond with just: "No change."

Respond with ONLY the sentence, no JSON, no formatting."""

_INVESTIGATE_PROMPT = """\
You are WatchTower's investigation assistant. You have access to a scene memory log — timestamped summaries of what a camera has observed over time.

Given the memory log and the user's question, provide a clear, concise answer. Reference specific times when relevant. If you can't answer from the available memory, say so.

## Scene Memory Log
{memory_log}

## Recent Alerts
{alert_log}

Answer the user's question based on the above context. Reference specific times using **HH:MM:SS** format (bold). Keep your answer to 2-4 sentences. If there is not enough memory to answer, say so honestly."""


class SceneMemory:
    def __init__(self, max_entries: int = 360) -> None:
        """Max entries: 360 = 3 hours at one entry per 30 seconds."""
        self._client = anthropic.AsyncAnthropicBedrock(
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
        )
        self._entries: list[MemoryEntry] = []
        self._max_entries = max_entries

    @property
    def entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    async def add_entry(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        recent_alerts: list[Alert],
        timestamp: float,
    ) -> MemoryEntry | None:
        """Summarize the current scene and add to memory."""
        if frame.size == 0:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not ok:
            return None

        b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        det_context = ""
        if detections:
            det_context = f"\nDetections: {', '.join(d.class_name for d in detections[:10])}"

        try:
            response = await self._client.messages.create(
                model=_BEDROCK_MODEL_FAST,
                max_tokens=100,
                system=_SUMMARIZE_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Describe this scene.{det_context}",
                        },
                    ],
                }],
            )

            summary = response.content[0].text.strip()
            alert_ids = [a.id for a in recent_alerts if abs(a.timestamp - timestamp) < 30]

            entry = MemoryEntry(
                timestamp=timestamp,
                summary=summary,
                detection_count=len(detections),
            )

            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries:]

            return entry

        except Exception as e:
            log.error("Memory entry creation failed: %s", e)
            return None

    def get_context(self, start_time: float = 0, end_time: float = 0) -> str:
        """Get formatted memory log for a time range."""
        if not self._entries:
            return "(No memory entries yet)"

        if end_time == 0:
            end_time = time.time()
        if start_time == 0:
            start_time = self._entries[0].timestamp

        filtered = [
            e for e in self._entries
            if start_time <= e.timestamp <= end_time
        ]

        if not filtered:
            return "(No memory entries in this time range)"

        lines: list[str] = []
        for e in filtered:
            t = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
            lines.append(f"[{t}] {e.summary}")

        return "\n".join(lines)

    async def investigate(
        self,
        question: str,
        recent_alerts: list[Alert],
        time_range_minutes: int = 30,
    ) -> str:
        """Answer a question about what happened using the memory log."""
        now = time.time()
        memory_log = self.get_context(
            start_time=now - (time_range_minutes * 60),
            end_time=now,
        )

        alert_lines: list[str] = []
        for a in recent_alerts[:20]:
            t = time.strftime("%H:%M:%S", time.localtime(a.timestamp))
            alert_lines.append(f"[{t}] {a.rule_name} ({a.severity}): {a.narration}")
        alert_log = "\n".join(alert_lines) if alert_lines else "(No recent alerts)"

        prompt = _INVESTIGATE_PROMPT.replace("{memory_log}", memory_log).replace("{alert_log}", alert_log)

        try:
            response = await self._client.messages.create(
                model=_BEDROCK_MODEL_SMART,
                max_tokens=300,
                system=prompt,
                messages=[{
                    "role": "user",
                    "content": question,
                }],
            )
            return response.content[0].text.strip()

        except Exception as e:
            log.error("Investigation failed: %s", e)
            return "Sorry, I couldn't process that question right now."
