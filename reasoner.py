"""Block 2: Multi-frame reasoning loop.

Async LLM thinking loop that runs every ~10 seconds, analyzing multiple
frames to understand sequences, intent, and context that the per-frame
boolean rule engine cannot capture.
"""
from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field

import anthropic
import cv2
import numpy as np

from models import Alert, Detection, Rule, Zone

log = logging.getLogger("watchtower.reasoner")

_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"

_SYSTEM_PROMPT = """\
You are WatchTower's scene reasoning engine. You analyze sequences of camera frames to understand what is happening over time.

You receive 3-5 frames captured over the last 10-30 seconds, along with:
- Current YOLO detections (objects, positions, poses)
- Active monitoring rules
- Recent alert history

Your job is to provide:
1. **observation**: A concise 1-2 sentence summary of what is happening RIGHT NOW
2. **concerns**: Any safety or security concerns (empty array if none)
3. **suggested_alerts**: Alerts that SHOULD fire based on your understanding but that the mechanical rule engine might miss (e.g., intent, suspicious behavior, sequences). Only suggest alerts for genuinely concerning situations.
4. **prediction**: What is likely to happen next (1 sentence, or empty string if nothing notable)

## Output format
Return ONLY valid JSON (no markdown):
{
  "observation": "...",
  "concerns": ["...", ...],
  "suggested_alerts": [
    {"reason": "...", "severity": "low|medium|high|critical"}
  ],
  "prediction": "..."
}

## Guidelines
- Be specific about people, objects, and their positions/actions
- Only flag genuine concerns, not routine activity
- suggested_alerts should be EMPTY most of the time. Only include them for situations that are clearly dangerous, illegal, or require immediate attention (e.g., a person falling, a fire, an intruder, a child in danger)
- Do NOT suggest alerts for: blinking, yawning, minor movements, adjusting position, looking at a phone, normal daily activities, or anything that is routine human behavior
- Keep observations factual and calm
- If nothing notable is happening, say so briefly and return empty concerns and suggested_alerts
"""


@dataclass
class Insight:
    observation: str = ""
    concerns: list[str] = field(default_factory=list)
    suggested_alerts: list[dict] = field(default_factory=list)
    prediction: str = ""


class Reasoner:
    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropicBedrock(
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
        )

    async def analyze(
        self,
        frames: list[tuple[np.ndarray, float]],
        detections: list[Detection],
        active_rules: list[Rule],
        active_zones: list[Zone],
        recent_alerts: list[Alert],
    ) -> Insight:
        """Analyze a sequence of frames and return an insight."""
        if not frames:
            return Insight(observation="No frames available.")

        # Encode frames as base64
        image_blocks: list[dict] = []
        for i, (frame, ts) in enumerate(frames):
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not ok:
                continue
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            image_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            })

        if not image_blocks:
            return Insight(observation="Failed to encode frames.")

        # Build context
        context_parts: list[str] = []

        if detections:
            det_summary = ", ".join(
                f"{d.class_name} ({d.confidence:.0%})" for d in detections[:10]
            )
            context_parts.append(f"Current detections: {det_summary}")

        if active_rules:
            rule_summary = ", ".join(
                f'"{r.name}" ({r.severity})' for r in active_rules if r.enabled
            )
            context_parts.append(f"Active rules: {rule_summary}")

        if active_zones:
            zone_summary = ", ".join(z.name for z in active_zones)
            context_parts.append(f"Zones: {zone_summary}")

        if recent_alerts:
            alert_summary = ", ".join(
                f'"{a.rule_name}" at {a.timestamp:.0f}' for a in recent_alerts[:5]
            )
            context_parts.append(f"Recent alerts: {alert_summary}")

        context_text = (
            f"These are {len(image_blocks)} frames from the last "
            f"{frames[-1][1] - frames[0][1]:.0f} seconds.\n\n"
            + "\n".join(context_parts)
        )

        content: list[dict] = image_blocks + [{"type": "text", "text": context_text}]

        try:
            response = await self._client.messages.create(
                model=_BEDROCK_MODEL,
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )

            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1])

            parsed = json.loads(raw)
            return Insight(
                observation=parsed.get("observation", ""),
                concerns=parsed.get("concerns", []),
                suggested_alerts=parsed.get("suggested_alerts", []),
                prediction=parsed.get("prediction", ""),
            )

        except Exception as e:
            log.error("Reasoning failed: %s", e)
            return Insight(observation="Reasoning temporarily unavailable.")
