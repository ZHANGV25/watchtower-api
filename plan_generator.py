from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic

from models import Condition, Rule

log = logging.getLogger("watchtower.plan_generator")

_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"

_SYSTEM_PROMPT = """You are WatchTower's monitoring AI. Analyze the user's input and respond appropriately.

## Classification
- If the input describes a SINGLE specific detection rule (e.g., "alert if someone enters the kitchen", "detect dogs", "notify me when a car appears"), respond with type "rule".
- If the input describes a BROADER SCENARIO or situation that implies multiple monitoring concerns (e.g., "my elderly mother is home alone", "overnight security for my shop", "baby in the nursery"), respond with type "scenario".

## Available condition types and their params:

1. object_present: {"class": "<yolo_class>"}
   YOLO classes: person, bicycle, car, motorcycle, bus, truck, boat, bird, cat, dog, horse, backpack, umbrella, handbag, suitcase, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, chair, couch, potted plant, bed, dining table, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, toothbrush

2. object_absent: {"class": "<yolo_class>"}

3. object_in_zone: {"class": "<yolo_class>", "zone": "<zone_name>"}

4. object_not_in_zone: {"class": "<yolo_class>", "zone": "<zone_name>"}

5. person_pose: {"pose": "standing"|"sitting"|"lying"|"crouching"}

6. count: {"class": "<yolo_class>", "operator": "gt"|"lt"|"gte"|"lte"|"eq", "value": <number>}

7. duration: {"seconds": <number>}
   Must be combined with other conditions. Means "conditions must be true for this many consecutive seconds."

8. time_window: {"start_hour": <0-23>, "end_hour": <0-23>}
   Rule only active during these hours.

## Available zones:
{zones}

## Response format

For a SINGLE RULE, respond with ONLY:
{"type": "rule", "rule": {"name": "<short name>", "conditions": [{"type": "<type>", "params": {<params>}}]}}

For a SCENARIO, respond with ONLY:
{"type": "scenario", "plan": {"name": "<plan name>", "description": "<one sentence>", "rules": [{"name": "<name>", "natural_language": "<what this watches for>", "conditions": [{"type": "<type>", "params": {<params>}}]}, ...]}}

## Guidelines for scenarios:
- Generate 3-7 rules that comprehensively cover the scenario
- If zones are available, use object_in_zone where relevant
- If no zones are available, use only non-zone conditions
- Include duration conditions for sustained states (e.g., person lying for >30 seconds implies a fall)
- Include time_window where appropriate (nighttime rules, work hours)
- Think about what could go wrong and create a rule for each concern
- Each rule should be independent and useful on its own
- If the user mentions "falling" or "fell", use person_pose with "lying" + duration
- Order rules from most to least important

Return ONLY valid JSON. No markdown, no explanation."""


class PlanGenerator:
    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropicBedrock(
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
        )

    async def classify_and_generate(
        self, text: str, zone_names: list[str],
    ) -> dict[str, Any] | None:
        """Classify input as rule or scenario and generate the appropriate output.

        Returns:
            {"type": "rule", "rule": Rule, "missing_zones": [...]} or
            {"type": "scenario", "plan": {...}} or
            None on failure.
        """
        zones_str = ", ".join(zone_names) if zone_names else "(no zones defined - use only non-zone conditions)"

        try:
            response = await self._client.messages.create(
                model=_BEDROCK_MODEL,
                max_tokens=2048,
                system=_SYSTEM_PROMPT.replace("{zones}", zones_str),
                messages=[{"role": "user", "content": text}],
            )

            raw = response.content[0].text.strip()

            # Strip markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)
            result_type = parsed.get("type", "rule")

            if result_type == "rule":
                rule_data = parsed.get("rule", {})
                conditions = [
                    Condition(type=c["type"], params=c.get("params", {}))
                    for c in rule_data.get("conditions", [])
                ]

                # Check for missing zones
                missing_zones = []
                for c in conditions:
                    if c.type in ("object_in_zone", "object_not_in_zone"):
                        ref = c.params.get("zone", "")
                        if ref and not any(z.lower() == ref.lower() for z in zone_names):
                            missing_zones.append(ref)

                rule = Rule(
                    name=rule_data.get("name", "Unnamed rule"),
                    natural_language=text,
                    conditions=conditions,
                    severity="medium",
                )

                return {"type": "rule", "rule": rule, "missing_zones": missing_zones}

            elif result_type == "scenario":
                plan_data = parsed.get("plan", {})
                rules = []
                for r in plan_data.get("rules", []):
                    conditions = [
                        Condition(type=c["type"], params=c.get("params", {}))
                        for c in r.get("conditions", [])
                    ]
                    rules.append(Rule(
                        name=r.get("name", "Unnamed rule"),
                        natural_language=r.get("natural_language", ""),
                        conditions=conditions,
                        severity="medium",
                    ))

                if not rules:
                    log.warning("Plan generated with 0 rules for: %s", text)
                    return None

                return {
                    "type": "scenario",
                    "plan": {
                        "name": plan_data.get("name", "Monitoring Plan"),
                        "description": plan_data.get("description", ""),
                        "scenario": text,
                        "rules": rules,
                    },
                }

            return None

        except Exception as e:
            log.error("Plan generation failed for '%s': %s", text, e)
            return None
