"""Per-camera processing session management.

Each connected camera gets its own CameraSession with independent
rule engine state, replay buffer, scene memory, and anomaly detector.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np
from fastapi import WebSocket

from anomaly import AnomalyDetector
from memory import SceneMemory
from models import Alert, Detection, Rule, Zone
from replay_buffer import ReplayBuffer
from rule_engine import RuleEngine

log = logging.getLogger("watchtower.camera_manager")


@dataclass
class CameraSession:
    """Runtime state for one connected camera."""
    camera_id: str
    name: str = ""

    # Camera connection
    camera_ws: WebSocket | None = None
    latest_frame: np.ndarray | None = None
    latest_detections: list[Detection] = field(default_factory=list)

    # Per-camera state (loaded from DB at start)
    zones: list[Zone] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)  # recent in-memory cache

    # Per-camera processing instances
    rule_engine: RuleEngine = field(default_factory=RuleEngine)
    replay_buffer: ReplayBuffer = field(default_factory=lambda: ReplayBuffer(max_seconds=1800, fps=2))
    scene_memory: SceneMemory = field(default_factory=SceneMemory)
    anomaly_detector: AnomalyDetector = field(default_factory=AnomalyDetector)

    # Feature toggles
    reasoning_enabled: bool = False
    narration_enabled: bool = False
    bootstrap_sent: bool = False

    # Processing task
    processing_task: asyncio.Task | None = field(default=None, repr=False)

    # Frontend viewers subscribed to this camera
    viewers: list[WebSocket] = field(default_factory=list)


class CameraManager:
    """Manages all camera sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, CameraSession] = {}

    @property
    def sessions(self) -> dict[str, CameraSession]:
        return self._sessions

    def get_session(self, camera_id: str) -> CameraSession | None:
        return self._sessions.get(camera_id)

    def get_or_create_session(self, camera_id: str, name: str = "") -> CameraSession:
        if camera_id not in self._sessions:
            self._sessions[camera_id] = CameraSession(
                camera_id=camera_id,
                name=name or camera_id,
            )
            log.info("Created camera session: %s", camera_id)
        return self._sessions[camera_id]

    def remove_session(self, camera_id: str) -> None:
        session = self._sessions.pop(camera_id, None)
        if session and session.processing_task:
            session.processing_task.cancel()
        if session:
            log.info("Removed camera session: %s", camera_id)

    def list_sessions(self) -> list[CameraSession]:
        return list(self._sessions.values())

    async def add_viewer(self, camera_id: str, ws: WebSocket) -> CameraSession | None:
        """Subscribe a frontend WebSocket to a camera's feed."""
        session = self._sessions.get(camera_id)
        if session and ws not in session.viewers:
            session.viewers.append(ws)
        return session

    async def remove_viewer(self, ws: WebSocket) -> None:
        """Remove a frontend WebSocket from all camera sessions."""
        for session in self._sessions.values():
            if ws in session.viewers:
                session.viewers.remove(ws)
