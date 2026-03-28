from __future__ import annotations

import json
import logging
import os

import anthropic

from models import Condition, Rule

log = logging.getLogger("watchtower.rule_parser")

_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"

_SYSTEM_PROMPT = """You are a camera monitoring rule compiler for WatchTower, a real-time surveillance system.

The user describes a monitoring rule in plain English. Translate it into a structured JSON rule.

## Available condition types and their params:

1. object_present: {"class": "<yolo_class>"}
   YOLO classes: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

2. object_absent: {"class": "<yolo_class>"}

3. object_in_zone: {"class": "<yolo_class>", "zone": "<zone_name>"}

4. object_not_in_zone: {"class": "<yolo_class>", "zone": "<zone_name>"}

5. person_size: {"size": "small"|"large", "threshold": 0.3}
   small = child-sized (bbox height < threshold of frame). large = adult-sized.

6. person_pose: {"pose": "standing"|"sitting"|"lying"|"crouching"}

7. count: {"class": "<yolo_class>", "operator": "gt"|"lt"|"gte"|"lte"|"eq", "value": <number>}

8. duration: {"seconds": <number>}
   Must be combined with other conditions. Means "the other conditions must be true for this many consecutive seconds."

9. time_window: {"start_hour": <0-23>, "end_hour": <0-23>}
   Rule only active during these hours.

## Available zones:
{zones}

## Output format:
Return ONLY valid JSON (no markdown, no explanation):
{{
  "name": "<short descriptive name>",
  "conditions": [<array of conditions>],
  "severity": "low"|"medium"|"high"|"critical"
}}

Each condition: {{"type": "<condition_type>", "params": {{...}}}}

## Guidelines:
- Use the simplest set of conditions that captures the intent
- If the user mentions a zone name that matches an available zone, use object_in_zone
- If the user mentions "child" or "kid", use person_size with size "small"
- If the user mentions "adult", use person_size with size "large"
- If the user mentions a time, use time_window
- If the user says "for more than X seconds/minutes", add a duration condition
- If the user mentions "falling", "fell", or "fall down", use person_pose with pose "lying" (there is no separate falling detector)
- If you cannot map the request to available conditions, approximate and reflect that in the name
- Always set severity to "medium" (the user will override it manually)"""


class RuleParser:
    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropicBedrock(
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
        )

    async def parse(self, text: str, zone_names: list[str], severity: str = "medium") -> tuple[Rule, list[str]] | None:
        zones_str = ", ".join(zone_names) if zone_names else "(no zones defined yet)"

        try:
            response = await self._client.messages.create(
                model=_BEDROCK_MODEL,
                max_tokens=1024,
                system=_SYSTEM_PROMPT.replace("{zones}", zones_str),
                messages=[{"role": "user", "content": text}],
            )

            raw = response.content[0].text.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)

            conditions = [
                Condition(type=c["type"], params=c.get("params", {}))
                for c in parsed.get("conditions", [])
            ]

            # Check for zone references that don't match any existing zone
            missing_zones = []
            for c in conditions:
                if c.type in ("object_in_zone", "object_not_in_zone"):
                    ref = c.params.get("zone", "")
                    if ref and not any(
                        z.lower() == ref.lower() for z in zone_names
                    ):
                        missing_zones.append(ref)

            rule = Rule(
                name=parsed.get("name", "Unnamed rule"),
                natural_language=text,
                conditions=conditions,
                severity=severity,
            )

            return rule, missing_zones

        except Exception as e:
            log.error("Failed to parse rule '%s': %s", text, e)
            return None
