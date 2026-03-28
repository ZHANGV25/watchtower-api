"""Integration tests for the plan generator - classification + generation."""
from __future__ import annotations

import sys
import os
import asyncio
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from plan_generator import PlanGenerator


@pytest.fixture
def pg():
    return PlanGenerator()


@pytest.mark.asyncio
class TestClassification:
    async def test_single_rule_classified(self, pg):
        """Simple detection request should be classified as a rule."""
        result = await pg.classify_and_generate("alert if a person is detected", [])
        assert result is not None
        assert result["type"] == "rule"
        assert result["rule"].name
        assert len(result["rule"].conditions) >= 1

    async def test_scenario_classified(self, pg):
        """Broad scenario should be classified as a scenario."""
        result = await pg.classify_and_generate(
            "my elderly mother is home alone",
            ["Kitchen", "Front Door"],
        )
        assert result is not None
        assert result["type"] == "scenario"
        plan = result["plan"]
        assert plan["name"]
        assert len(plan["rules"]) >= 3

    async def test_another_single_rule(self, pg):
        """Specific object detection should be a rule."""
        result = await pg.classify_and_generate("detect dogs in the yard", [])
        assert result is not None
        assert result["type"] == "rule"

    async def test_another_scenario(self, pg):
        """Store security scenario should be a scenario."""
        result = await pg.classify_and_generate(
            "overnight security for my shop",
            ["Front Door", "Cash Register", "Storage Room"],
        )
        assert result is not None
        assert result["type"] == "scenario"
        assert len(result["plan"]["rules"]) >= 3


@pytest.mark.asyncio
class TestPlanGeneration:
    async def test_plan_uses_available_zones(self, pg):
        """Plan rules should reference zones when available."""
        result = await pg.classify_and_generate(
            "baby in the nursery while I'm cooking",
            ["Nursery", "Kitchen", "Front Door"],
        )
        assert result is not None
        assert result["type"] == "scenario"
        plan = result["plan"]

        # At least one rule should use object_in_zone
        all_types = [c.type for r in plan["rules"] for c in r.conditions]
        assert "object_in_zone" in all_types, f"Expected zone-aware rules, got {all_types}"

    async def test_plan_without_zones(self, pg):
        """Plan should work without zones, using non-zone conditions only."""
        result = await pg.classify_and_generate(
            "keeping my toddler safe at home",
            [],
        )
        assert result is not None
        assert result["type"] == "scenario"
        plan = result["plan"]
        assert len(plan["rules"]) >= 3

        # Should not have zone conditions since none available
        all_types = [c.type for r in plan["rules"] for c in r.conditions]
        assert "object_in_zone" not in all_types

    async def test_plan_has_descriptions(self, pg):
        """Each rule in the plan should have a natural_language description."""
        result = await pg.classify_and_generate(
            "monitoring an elderly person who lives alone",
            ["Bedroom", "Bathroom"],
        )
        assert result is not None
        assert result["type"] == "scenario"
        for rule in result["plan"]["rules"]:
            assert rule.natural_language, f"Rule '{rule.name}' missing natural_language"

    async def test_plan_has_name_and_description(self, pg):
        """Plan should have a name and description."""
        result = await pg.classify_and_generate(
            "I'm leaving my dog alone in the apartment",
            [],
        )
        assert result is not None
        if result["type"] == "scenario":
            assert result["plan"]["name"]
            assert result["plan"]["description"]


@pytest.mark.asyncio
class TestWebSocketPlanFlow:
    """Test the full WebSocket flow: generate_plan -> plan_generated -> apply_plan."""

    async def test_scenario_generates_plan(self):
        """Sending generate_plan with a scenario should receive plan_generated."""
        import websockets

        async with websockets.connect("ws://localhost:8000/ws") as ws:
            # Consume until init
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                if json.loads(raw)["type"] == "init":
                    break

            # Send scenario
            await ws.send(json.dumps({
                "type": "generate_plan",
                "payload": {"text": "overnight home security while I sleep"},
            }))

            # Wait for plan_generated or rule_added
            deadline = asyncio.get_event_loop().time() + 30
            while asyncio.get_event_loop().time() < deadline:
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                msg = json.loads(raw)
                if msg["type"] == "plan_generated":
                    plan = msg["payload"]
                    assert "id" in plan
                    assert "name" in plan
                    assert "rules" in plan
                    assert len(plan["rules"]) >= 2
                    return
                if msg["type"] == "rule_added":
                    # Classified as single rule - acceptable
                    return

            pytest.fail("Never received plan_generated or rule_added")

    async def test_single_rule_via_generate_plan(self):
        """Sending generate_plan with a simple rule should receive rule_added."""
        import websockets

        async with websockets.connect("ws://localhost:8000/ws") as ws:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                if json.loads(raw)["type"] == "init":
                    break

            await ws.send(json.dumps({
                "type": "generate_plan",
                "payload": {"text": "alert if a cat appears"},
            }))

            deadline = asyncio.get_event_loop().time() + 30
            while asyncio.get_event_loop().time() < deadline:
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                msg = json.loads(raw)
                if msg["type"] == "rule_added":
                    assert msg["payload"]["name"]
                    return
                if msg["type"] == "plan_generated":
                    # Classified as scenario - also acceptable
                    return

            pytest.fail("Never received rule_added or plan_generated")
