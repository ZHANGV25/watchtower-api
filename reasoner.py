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
You are WatchTower's elder care scene reasoning engine. You analyze sequences of camera frames to understand what an elderly person is doing and ensure their safety.

You receive 3-5 frames captured over the last 10-30 seconds, along with:
- Current YOLO detections (objects, positions, poses)
- Active monitoring rules
- Recent alert history

Your job is to provide:
1. **observation**: A concise 1-2 sentence summary of what is happening RIGHT NOW
2. **concerns**: Any safety or health concerns (empty array if none)
3. **suggested_alerts**: Alerts that SHOULD fire based on your understanding but that the mechanical rule engine might miss. Only suggest alerts for genuinely concerning situations.
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

## Elder Care Guidelines
- **Sleep vs Fall**: If a person is lying down on a bed or couch during nighttime hours (10pm-7am), they are likely sleeping — NOT a fall. If a person is lying on the FLOOR, especially during daytime, this is likely a fall and is CRITICAL.
- **Mobility**: Note if the person appears unsteady, moving slowly, using furniture for support, or having difficulty standing up. These are important mobility observations.
- **Confusion signs**: Watch for wandering aimlessly, standing still for long periods looking confused, repeatedly going to the same spot, or seeming disoriented.
- **Meals and hydration**: Note when the person goes to the kitchen, sits at a table, handles cups/dishes/food. These are important ADL (Activities of Daily Living) observations.
- **Visitor interactions**: Note if a visitor is present, how the elderly person engages with them (active conversation vs withdrawn/passive).
- **Routine activity**: Normal activities like watching TV, reading, eating, walking between rooms are GOOD signs. Note them positively.
- **Cross-room context**: This camera monitors ONE room. The person may be in another room when not visible here. Absence from this camera does NOT mean the person is inactive - they may simply be in another part of the house. Only flag inactivity if you are told there is no activity across ALL rooms.
- **Person identification**: In a single-elder household, when only ONE person is detected, that is the elderly resident being monitored. When TWO or more people are detected, the resident is present along with a visitor or caregiver. Do not count visitors as the primary resident for activity tracking. Note visitor arrivals and departures separately.

## Alert Guidelines
- suggested_alerts should be EMPTY most of the time. Only include them for situations that are clearly dangerous or require immediate attention.
- CRITICAL: Person on the floor, no movement for extended period, signs of medical emergency
- HIGH: Unsteady movement suggesting fall risk, signs of confusion, missed medication time
- MEDIUM: Unusual nighttime activity, prolonged inactivity during daytime
- LOW: Visitor arrived/departed
- Do NOT alert for: normal sleeping, watching TV, sitting quietly, minor position adjustments, routine daily activities
- Keep observations factual, warm, and calm — family members will see these
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
            det_labels = []
            for d in detections[:10]:
                label = f"{d.identity} ({d.class_name})" if d.identity else d.class_name
                det_labels.append(f"{label} ({d.confidence:.0%})")
            det_summary = ", ".join(det_labels)
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
