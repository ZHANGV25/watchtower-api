"""Tests for Block 1: Scene Analyzer."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scene_analyzer import SceneAnalysis, SceneAnalyzer


class TestSceneAnalysis:
    def test_default_values(self):
        sa = SceneAnalysis()
        assert sa.scene_type == ""
        assert sa.scene_description == ""
        assert sa.zones == []
        assert sa.suggested_rules == []

    def test_custom_values(self):
        sa = SceneAnalysis(
            scene_type="home",
            scene_description="A living room",
            zones=[{"name": "Door", "x": 10, "y": 20, "width": 30, "height": 40}],
            suggested_rules=[{
                "name": "Fall detection",
                "natural_language": "Alert if someone falls",
                "conditions": [{"type": "person_pose", "params": {"pose": "lying"}}],
                "severity": "critical",
            }],
        )
        assert sa.scene_type == "home"
        assert len(sa.zones) == 1
        assert sa.zones[0]["name"] == "Door"
        assert len(sa.suggested_rules) == 1


class TestSceneAnalyzer:
    def test_init(self):
        analyzer = SceneAnalyzer()
        assert analyzer._client is not None

    def test_analyze_empty_frame(self):
        """An empty/invalid frame should return empty SceneAnalysis."""
        analyzer = SceneAnalyzer()
        # Create a 0x0 frame that will fail encoding
        frame = np.array([], dtype=np.uint8)
        import asyncio
        result = asyncio.run(analyzer.analyze(frame))
        assert isinstance(result, SceneAnalysis)
        assert result.scene_type == ""


@pytest.mark.asyncio
async def test_analyze_valid_frame_mock(monkeypatch):
    """Test scene analysis with a mocked LLM response."""
    mock_response = json.dumps({
        "scene_type": "home",
        "scene_description": "A kitchen with a counter and door",
        "zones": [
            {"name": "Counter", "x": 10, "y": 30, "width": 40, "height": 20},
            {"name": "Door", "x": 70, "y": 10, "width": 20, "height": 60},
        ],
        "suggested_rules": [
            {
                "name": "Fall near counter",
                "natural_language": "Alert if person falls near counter",
                "conditions": [
                    {"type": "person_pose", "params": {"pose": "lying"}},
                    {"type": "object_in_zone", "params": {"class": "person", "zone": "Counter"}},
                ],
                "severity": "critical",
            },
        ],
    })

    class MockContent:
        text = mock_response

    class MockResponse:
        content = [MockContent()]

    async def mock_create(**kwargs):
        return MockResponse()

    analyzer = SceneAnalyzer()
    monkeypatch.setattr(analyzer._client.messages, "create", mock_create)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = await analyzer.analyze(frame)

    assert result.scene_type == "home"
    assert "kitchen" in result.scene_description.lower()
    assert len(result.zones) == 2
    assert result.zones[0]["name"] == "Counter"
    assert len(result.suggested_rules) == 1
    assert result.suggested_rules[0]["severity"] == "critical"
