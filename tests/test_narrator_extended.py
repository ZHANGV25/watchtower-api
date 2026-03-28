"""Tests for Block 5: Extended narrator with live narration."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models import BBox, Detection
from narrator import Narrator


class TestNarrateScene:
    @pytest.mark.asyncio
    async def test_narrate_scene_invalid_frame(self):
        """Invalid frame should return empty string."""
        n = Narrator()
        frame = np.array([], dtype=np.uint8)
        result = await n.narrate_scene(frame)
        assert result == ""

    @pytest.mark.asyncio
    async def test_narrate_scene_mock(self, monkeypatch):
        """Test scene narration with mocked LLM."""
        class MockContent:
            text = "A person is sitting on the couch reading a book."

        class MockResponse:
            content = [MockContent()]

        async def mock_create(**kwargs):
            return MockResponse()

        n = Narrator()
        monkeypatch.setattr(n._client.messages, "create", mock_create)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(class_name="person", confidence=0.9, bbox=BBox(x=30, y=20, width=15, height=40)),
            Detection(class_name="book", confidence=0.8, bbox=BBox(x=35, y=30, width=5, height=8)),
        ]
        result = await n.narrate_scene(frame, detections)

        assert "person" in result.lower()
        assert "couch" in result.lower()

    @pytest.mark.asyncio
    async def test_narrate_scene_without_detections(self, monkeypatch):
        """Narration should work without detection context."""
        class MockContent:
            text = "An empty room with a window."

        class MockResponse:
            content = [MockContent()]

        async def mock_create(**kwargs):
            return MockResponse()

        n = Narrator()
        monkeypatch.setattr(n._client.messages, "create", mock_create)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = await n.narrate_scene(frame)

        assert "room" in result.lower()

    @pytest.mark.asyncio
    async def test_narrate_scene_error_returns_empty(self, monkeypatch):
        """LLM error should return empty string, not crash."""
        async def mock_create(**kwargs):
            raise Exception("API error")

        n = Narrator()
        monkeypatch.setattr(n._client.messages, "create", mock_create)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = await n.narrate_scene(frame)
        assert result == ""
