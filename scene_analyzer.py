"""Block 1: Self-bootstrapping scene analysis.

Combines zone detection and rule suggestion into a single LLM call.
Camera turns on → analyzes scene → suggests zones + rules → user confirms.
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

log = logging.getLogger("watchtower.scene_analyzer")

_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"

_SYSTEM_PROMPT = """\
You are WatchTower's scene analyzer. Given a camera frame, you must:

1. Identify what kind of space this is (home, office, retail, warehouse, etc.)
2. Identify distinct zones (areas of interest) with bounding boxes
3. Suggest practical monitoring rules for this specific space

## Zone format
Each zone: {"name": "descriptive label", "x": 0, "y": 0, "width": 0, "height": 0}
- Coordinates are percentages of frame dimensions (0-100)
- Be conservative: 3-8 zones, no overlaps
- Focus on: doors, stairs, windows, furniture, hazards, entry/exit points

## Rule format
Each rule has conditions using these primitives:
- object_present: {"class": "<yolo_class>"} — YOLO classes include: person, car, dog, cat, chair, couch, bed, tv, laptop, cell phone, bottle, cup, book, etc.
- object_absent: {"class": "<yolo_class>"}
- object_in_zone: {"class": "<yolo_class>", "zone": "<zone_name>"}
- object_not_in_zone: {"class": "<yolo_class>", "zone": "<zone_name>"}
- person_size: {"size": "small"|"large", "threshold": 0.3}
- person_pose: {"pose": "standing"|"sitting"|"lying"|"crouching"}
- count: {"class": "<yolo_class>", "operator": "gt"|"lt"|"gte"|"lte"|"eq", "value": N}
- duration: {"seconds": N} — other conditions must hold for N consecutive seconds
- time_window: {"start_hour": 0-23, "end_hour": 0-23}

## Output format
Return ONLY valid JSON (no markdown):
{
  "scene_type": "home|office|retail|warehouse|outdoor|other",
  "scene_description": "One sentence describing what you see",
  "zones": [{"name": "...", "x": 0, "y": 0, "width": 0, "height": 0}, ...],
  "suggested_rules": [
    {
      "name": "short name",
      "natural_language": "human-readable description",
      "conditions": [{"type": "...", "params": {...}}, ...],
      "severity": "low|medium|high|critical"
    },
    ...
  ]
}

## Guidelines
- Suggest 3-5 rules that are practical for the space type
- For homes: focus on safety (falls, doors, inactivity)
- For offices: focus on occupancy, after-hours access
- For retail: focus on crowd, restricted areas
- Rules should reference the zones you identified by name
- Always include at least one fall detection rule for spaces with people
"""


@dataclass
class SceneAnalysis:
    scene_type: str = ""
    scene_description: str = ""
    zones: list[dict] = field(default_factory=list)
    suggested_rules: list[dict] = field(default_factory=list)


class SceneAnalyzer:
    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropicBedrock(
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
        )

    async def analyze(self, frame: np.ndarray) -> SceneAnalysis:
        """Analyze a camera frame and return zones + suggested rules."""
        if frame.size == 0:
            return SceneAnalysis()
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return SceneAnalysis()

        b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        try:
            response = await self._client.messages.create(
                model=_BEDROCK_MODEL,
                max_tokens=2048,
                system=_SYSTEM_PROMPT,
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
                            "text": "Analyze this scene. Identify zones and suggest monitoring rules.",
                        },
                    ],
                }],
            )

            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1])

            parsed = json.loads(raw)
            return SceneAnalysis(
                scene_type=parsed.get("scene_type", "other"),
                scene_description=parsed.get("scene_description", ""),
                zones=parsed.get("zones", []),
                suggested_rules=parsed.get("suggested_rules", []),
            )

        except Exception as e:
            log.error("Scene analysis failed: %s", e)
            return SceneAnalysis()
