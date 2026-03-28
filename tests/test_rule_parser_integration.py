"""Integration tests for rule parsing via Bedrock.

These tests call the actual Bedrock API to verify the LLM produces valid rules.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

from rule_parser import RuleParser


@pytest.fixture
def parser():
    return RuleParser()


@pytest.mark.asyncio
class TestRuleParserBedrock:
    async def test_simple_person_detection(self, parser):
        """'Alert if a person is detected' should produce an object_present rule."""
        result = await parser.parse("Alert me if a person is detected", [])
        assert result is not None, "Parser returned None -- Bedrock call likely failed"
        rule, missing = result
        assert rule.name, "Rule should have a name"
        assert len(rule.conditions) >= 1
        types = [c.type for c in rule.conditions]
        assert "object_present" in types, f"Expected object_present, got {types}"
        person_cond = next(c for c in rule.conditions if c.type == "object_present")
        assert person_cond.params.get("class") == "person"

    async def test_zone_aware_rule(self, parser):
        """Rule mentioning a zone should use object_in_zone."""
        result = await parser.parse(
            "Alert if a person enters the Kitchen",
            ["Kitchen", "Front Door"],
        )
        assert result is not None
        rule, missing = result
        types = [c.type for c in rule.conditions]
        assert "object_in_zone" in types, f"Expected object_in_zone, got {types}"
        assert len(missing) == 0, "Kitchen zone exists, should not be missing"

    async def test_missing_zone_warning(self, parser):
        """Rule referencing a zone that doesn't exist should report it."""
        result = await parser.parse(
            "Alert if a person enters the Garage",
            ["Kitchen"],
        )
        assert result is not None
        rule, missing = result
        assert len(missing) > 0, "Garage zone doesn't exist, should be flagged"
        assert any("garage" in z.lower() for z in missing)

    async def test_count_rule(self, parser):
        """Rule about number of people should produce a count condition."""
        result = await parser.parse("Alert if more than 3 people are in the room", [])
        assert result is not None
        rule, _ = result
        types = [c.type for c in rule.conditions]
        assert "count" in types, f"Expected count condition, got {types}"

    async def test_severity_override(self, parser):
        """Severity should come from the caller, not the LLM."""
        result = await parser.parse(
            "Alert if a person is detected",
            [],
            severity="critical",
        )
        assert result is not None
        rule, _ = result
        assert rule.severity == "critical"

    async def test_invalid_input_handled(self, parser):
        """Nonsense input should still return a result or None, not crash."""
        result = await parser.parse("asdfghjkl random gibberish 12345", [])
        if result is not None:
            rule, _ = result
            assert rule.name
