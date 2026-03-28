"""Tests for Block 2: Reasoner."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import Alert, BBox, Detection, Rule, Condition, Zone
from reasoner import Insight, Reasoner


class TestInsight:
    def test_default_values(self):
        insight = Insight()
        assert insight.observation == ""
        assert insight.concerns == []
        assert insight.suggested_alerts == []
        assert insight.prediction == ""

    def test_custom_values(self):
        insight = Insight(
            observation="Person walking to door",
            concerns=["Approaching restricted area"],
            suggested_alerts=[{"reason": "Suspicious approach", "severity": "high"}],
            prediction="Person may enter restricted area",
        )
        assert "walking" in insight.observation
        assert len(insight.concerns) == 1
        assert len(insight.suggested_alerts) == 1
        assert insight.suggested_alerts[0]["severity"] == "high"


class TestReasoner:
    def test_init(self):
        r = Reasoner()
        assert r._client is not None

    @pytest.mark.asyncio
    async def test_analyze_no_frames(self):
        r = Reasoner()
        result = await r.analyze([], [], [], [], [])
        assert isinstance(result, Insight)
        assert "No frames" in result.observation


@pytest.mark.asyncio
async def test_analyze_with_mock(monkeypatch):
    """Test reasoning with mocked LLM response."""
    mock_response = json.dumps({
        "observation": "One person sitting on a couch watching TV.",
        "concerns": [],
        "suggested_alerts": [],
        "prediction": "No immediate changes expected.",
    })

    class MockContent:
        text = mock_response

    class MockResponse:
        content = [MockContent()]

    async def mock_create(**kwargs):
        return MockResponse()

    r = Reasoner()
    monkeypatch.setattr(r._client.messages, "create", mock_create)

    frames = [
        (np.zeros((480, 640, 3), dtype=np.uint8), 1000.0),
        (np.zeros((480, 640, 3), dtype=np.uint8), 1005.0),
        (np.zeros((480, 640, 3), dtype=np.uint8), 1010.0),
    ]
    detections = [
        Detection(
            class_name="person",
            confidence=0.95,
            bbox=BBox(x=30, y=20, width=15, height=40),
        )
    ]
    rules_list = [
        Rule(
            name="Fall detection",
            natural_language="Alert if someone falls",
            conditions=[Condition(type="person_pose", params={"pose": "lying"})],
            severity="critical",
        )
    ]
    zones_list = [Zone(name="Living Room", x=0, y=0, width=100, height=100)]

    result = await r.analyze(frames, detections, rules_list, zones_list, [])

    assert "couch" in result.observation.lower()
    assert result.concerns == []
    assert result.suggested_alerts == []


@pytest.mark.asyncio
async def test_analyze_with_concerns_mock(monkeypatch):
    """Test reasoning that produces concerns and suggested alerts."""
    mock_response = json.dumps({
        "observation": "Person lying motionless on floor for extended period.",
        "concerns": ["Person may have fallen", "No movement detected"],
        "suggested_alerts": [
            {"reason": "Possible fall detected by reasoning", "severity": "critical"}
        ],
        "prediction": "Medical attention may be needed.",
    })

    class MockContent:
        text = mock_response

    class MockResponse:
        content = [MockContent()]

    async def mock_create(**kwargs):
        return MockResponse()

    r = Reasoner()
    monkeypatch.setattr(r._client.messages, "create", mock_create)

    frames = [(np.zeros((480, 640, 3), dtype=np.uint8), 1000.0)]
    result = await r.analyze(frames, [], [], [], [])

    assert len(result.concerns) == 2
    assert len(result.suggested_alerts) == 1
    assert result.suggested_alerts[0]["severity"] == "critical"
    assert "Medical" in result.prediction
