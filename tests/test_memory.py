"""Tests for Block 4: Scene Memory."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from memory import MemoryEntry, SceneMemory
from models import Alert, BBox, Detection


class TestMemoryEntry:
    def test_default_values(self):
        entry = MemoryEntry(timestamp=1000.0, summary="A person standing.")
        assert entry.detection_count == 0

    def test_custom_values(self):
        entry = MemoryEntry(
            timestamp=1000.0,
            summary="Two people in kitchen",
            detection_count=2,
        )
        assert entry.detection_count == 2


class TestSceneMemory:
    def test_init(self):
        mem = SceneMemory()
        assert len(mem.entries) == 0

    def test_get_context_empty(self):
        mem = SceneMemory()
        ctx = mem.get_context()
        assert "No memory entries yet" in ctx

    def test_get_context_with_entries(self):
        mem = SceneMemory()
        now = time.time()
        mem._entries.append(MemoryEntry(timestamp=now - 60, summary="Person entered room"))
        mem._entries.append(MemoryEntry(timestamp=now - 30, summary="Person sat on couch"))
        mem._entries.append(MemoryEntry(timestamp=now, summary="Person watching TV"))

        ctx = mem.get_context()
        assert "entered room" in ctx
        assert "sat on couch" in ctx
        assert "watching TV" in ctx

    def test_get_context_time_range(self):
        mem = SceneMemory()
        now = time.time()
        mem._entries.append(MemoryEntry(timestamp=now - 120, summary="Old event"))
        mem._entries.append(MemoryEntry(timestamp=now - 30, summary="Recent event"))

        # Only get last 60 seconds
        ctx = mem.get_context(start_time=now - 60, end_time=now)
        assert "Old event" not in ctx
        assert "Recent event" in ctx

    def test_max_entries_limit(self):
        mem = SceneMemory(max_entries=5)
        now = time.time()
        for i in range(10):
            mem._entries.append(MemoryEntry(timestamp=now + i, summary=f"Event {i}"))
            if len(mem._entries) > mem._max_entries:
                mem._entries = mem._entries[-mem._max_entries:]

        assert len(mem._entries) == 5
        assert mem._entries[0].summary == "Event 5"

    @pytest.mark.asyncio
    async def test_add_entry_mock(self, monkeypatch):
        """Test adding a memory entry with mocked LLM."""
        class MockContent:
            text = "A person is standing near the kitchen counter."

        class MockResponse:
            content = [MockContent()]

        async def mock_create(**kwargs):
            return MockResponse()

        mem = SceneMemory()
        monkeypatch.setattr(mem._client.messages, "create", mock_create)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [Detection(
            class_name="person", confidence=0.9,
            bbox=BBox(x=30, y=20, width=15, height=40),
        )]

        entry = await mem.add_entry(frame, detections, [], time.time())

        assert entry is not None
        assert "kitchen" in entry.summary.lower()
        assert entry.detection_count == 1
        assert len(mem.entries) == 1

    @pytest.mark.asyncio
    async def test_investigate_mock(self, monkeypatch):
        """Test investigation with mocked LLM."""
        class MockContent:
            text = "A person entered the room at 14:32 and sat on the couch."

        class MockResponse:
            content = [MockContent()]

        async def mock_create(**kwargs):
            return MockResponse()

        mem = SceneMemory()
        monkeypatch.setattr(mem._client.messages, "create", mock_create)

        # Add some memory entries
        now = time.time()
        mem._entries.append(MemoryEntry(timestamp=now - 60, summary="Person entered"))
        mem._entries.append(MemoryEntry(timestamp=now - 30, summary="Person sat down"))

        answer = await mem.investigate(
            question="What happened recently?",
            recent_alerts=[],
        )

        assert "entered" in answer.lower()
        assert "couch" in answer.lower()

    @pytest.mark.asyncio
    async def test_add_entry_invalid_frame(self):
        """Invalid frame should return None."""
        mem = SceneMemory()
        frame = np.array([], dtype=np.uint8)
        entry = await mem.add_entry(frame, [], [], time.time())
        assert entry is None
