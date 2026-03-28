from __future__ import annotations

import base64
import json
import logging
import os

import anthropic
import cv2
import numpy as np

from models import Zone

log = logging.getLogger("watchtower.zone_generator")

_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"

_SYSTEM_PROMPT = """You are analyzing a camera frame to identify areas of interest for a safety monitoring system.

Look at the image and identify distinct areas that would be relevant for monitoring rules. For example:
- Doors and entryways
- Stairs or steps
- Kitchen areas (counters, stove, sink)
- Windows
- Furniture (couch, bed, chair, table)
- Hallways or corridors
- Open floor areas
- Any potentially hazardous areas

For each area, provide:
- name: a short descriptive label (e.g., "Front Door", "Staircase", "Kitchen Counter")
- x, y, width, height: bounding box as percentages of frame dimensions (0-100)

Return ONLY a JSON array (no markdown, no explanation):
[
  {"name": "...", "x": 0, "y": 0, "width": 0, "height": 0},
  ...
]

Be conservative. Only identify clearly visible and distinct areas. Typically 3-8 zones for a room.
Do not create overlapping zones. Each zone should be a distinct area."""


class ZoneGenerator:
    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropicBedrock(
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
        )

    async def generate(self, frame: np.ndarray) -> list[Zone]:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return []

        b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        try:
            response = await self._client.messages.create(
                model=_BEDROCK_MODEL,
                max_tokens=1024,
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
                            "text": "Identify the zones in this camera frame.",
                        },
                    ],
                }],
            )

            raw = response.content[0].text.strip()

            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1])

            parsed = json.loads(raw)

            colors = [
                "#22d3ee", "#a78bfa", "#34d399", "#fb923c",
                "#f472b6", "#facc15", "#60a5fa", "#e879f9",
            ]

            zones: list[Zone] = []
            for i, item in enumerate(parsed):
                zones.append(Zone(
                    name=item["name"],
                    x=float(item["x"]),
                    y=float(item["y"]),
                    width=float(item["width"]),
                    height=float(item["height"]),
                    color=colors[i % len(colors)],
                ))
            return zones

        except Exception as e:
            log.error("Zone generation failed: %s", e)
            return []
