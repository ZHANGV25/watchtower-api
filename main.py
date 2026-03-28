from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from actions import ActionEngine
from anomaly import AnomalyDetector, AnomalyPhase
from detector import Detector
from memory import SceneMemory
from models import Alert, MonitoringPlan, Rule, WSMessage, Zone
from narrator import Narrator
from plan_generator import PlanGenerator
from reasoner import Reasoner
from replay_buffer import ReplayBuffer
from rule_engine import RuleEngine
from rule_parser import RuleParser
from scene_analyzer import SceneAnalyzer
from zone_generator import ZoneGenerator

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("watchtower")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

detector = Detector()
rule_engine = RuleEngine()
rule_parser = RuleParser()
plan_generator = PlanGenerator()
zone_generator = ZoneGenerator()
narrator = Narrator()
replay_buffer = ReplayBuffer(max_seconds=1800, fps=2)
scene_analyzer = SceneAnalyzer()
reasoner = Reasoner()
action_engine = ActionEngine()
scene_memory = SceneMemory()
anomaly_detector = AnomalyDetector()

zones: list[Zone] = []
rules: list[Rule] = []
alerts: list[Alert] = []
pending_plans: dict[str, MonitoringPlan] = {}
connected_clients: list[WebSocket] = []

camera: cv2.VideoCapture | None = None
camera_running = False
camera_source: str | int = int(os.getenv("WATCHTOWER_CAMERA", "0"))  # index or URL

# Remote camera frame buffer (pushed by /ws/camera clients)
remote_frame: np.ndarray | None = None
remote_camera_connected = False

# Feature toggles
reasoning_enabled = False
narration_enabled = False
bootstrap_sent = False

# Latest state for new features
latest_detections: list = []
latest_frame: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def frame_to_b64(frame: np.ndarray, quality: int = 70) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


async def broadcast(msg: WSMessage) -> None:
    raw = msg.model_dump_json()
    dead: list[WebSocket] = []
    for ws in connected_clients:
        try:
            await ws.send_text(raw)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_clients.remove(ws)


async def broadcast_event(event_type: str, payload: dict[str, Any]) -> None:
    """Helper for action engine callbacks."""
    await broadcast(WSMessage(type=event_type, payload=payload))


# ---------------------------------------------------------------------------
# Camera loop
# ---------------------------------------------------------------------------

async def _get_next_frame() -> np.ndarray | None:
    """Get next frame from local camera or remote camera buffer."""
    global remote_frame
    if remote_camera_connected and remote_frame is not None:
        frame = remote_frame
        remote_frame = None  # consume it
        return frame
    if camera is not None and camera.isOpened():
        ok, frame = camera.read()
        return frame if ok else None
    return None


async def camera_loop() -> None:
    global camera, camera_running, bootstrap_sent, latest_detections, latest_frame

    # Try local camera if no remote camera is connected
    use_local = os.getenv("WATCHTOWER_NO_CAMERA") != "1" and not remote_camera_connected
    if use_local:
        source = camera_source
        # Try parsing as URL string
        if isinstance(source, str) and not source.isdigit():
            camera = cv2.VideoCapture(source)
        else:
            camera = cv2.VideoCapture(int(source) if isinstance(source, str) else source)
        if not camera.isOpened():
            log.warning("No local camera at source %s — waiting for remote camera", source)
            camera = None

    camera_running = True
    if camera is not None:
        log.info("Camera started (local source: %s)", camera_source)
    else:
        log.info("Camera loop started — waiting for remote camera feed")

    frame_interval = 1.0 / 24  # target ~24 fps
    last_frame_time = 0.0
    first_frame_captured = False

    while camera_running:
        now = time.time()
        if now - last_frame_time < frame_interval:
            await asyncio.sleep(0.003)
            continue

        frame = await _get_next_frame()
        if frame is None:
            await asyncio.sleep(0.05)
            continue

        last_frame_time = now
        latest_frame = frame

        # Auto-bootstrap on first frame (Block 1)
        if not first_frame_captured and not bootstrap_sent:
            first_frame_captured = True
            asyncio.create_task(_auto_bootstrap(frame))

        # Only run pose estimation if any rule uses pose conditions
        pose_types = {"person_pose", "person_falling"}
        need_pose = any(
            c.type in pose_types
            for r in rules if r.enabled
            for c in r.conditions
        )

        # Run detection in thread pool so it doesn't block the event loop
        loop = asyncio.get_event_loop()
        detections = await loop.run_in_executor(
            None, detector.detect, frame, need_pose
        )
        latest_detections = detections

        # Store in replay buffer
        replay_buffer.add_frame(frame, now)

        # Anomaly detection (Block 6)
        if anomaly_detector.phase == AnomalyPhase.LEARNING:
            done = anomaly_detector.learn_frame(frame)
            if done:
                await broadcast(WSMessage(type="anomaly_status", payload={
                    "phase": "detecting",
                    "time_remaining": 0,
                }))
        elif anomaly_detector.phase == AnomalyPhase.DETECTING:
            # Check every 2 seconds (not every frame)
            if not hasattr(camera_loop, "_last_anomaly") or now - camera_loop._last_anomaly > 2.0:  # type: ignore[attr-defined]
                camera_loop._last_anomaly = now  # type: ignore[attr-defined]
                score = anomaly_detector.detect(frame)
                await broadcast(WSMessage(type="anomaly_score", payload={
                    "score": round(score, 3),
                    "timestamp": now,
                }))
                # Cooldown: only fire anomaly alert every 30 seconds
                last_anomaly_alert = getattr(camera_loop, "_last_anomaly_alert", 0.0)
                if score > anomaly_detector.threshold and now - last_anomaly_alert > 30.0:
                    camera_loop._last_anomaly_alert = now  # type: ignore[attr-defined]
                    asyncio.create_task(_handle_anomaly_alert(frame, score, now))

        # Check rules
        fired = rule_engine.evaluate(rules, zones, detections, now)

        # Process fired alerts: verify with LLM before broadcasting
        for alert in fired:
            alert.frame_b64 = frame_to_b64(frame, quality=80)
            asyncio.create_task(_verify_and_broadcast_alert(alert, frame))

        # Encode frame off the event loop
        frame_b64 = await loop.run_in_executor(
            None, frame_to_b64, frame, 50
        )

        # Broadcast frame + detections
        await broadcast(WSMessage(
            type="frame",
            payload={
                "frame": frame_b64,
                "detections": [d.model_dump() for d in detections],
                "timestamp": now,
                "fps": round(1.0 / max(time.time() - now, 0.001)),
            },
        ))

    camera.release()
    log.info("Camera stopped")


# ---------------------------------------------------------------------------
# Block 1: Auto-bootstrap
# ---------------------------------------------------------------------------

async def _auto_bootstrap(frame: np.ndarray) -> None:
    global bootstrap_sent
    bootstrap_sent = True
    log.info("Running auto-bootstrap scene analysis...")

    analysis = await scene_analyzer.analyze(frame)
    if not analysis.scene_description:
        log.warning("Scene analysis returned empty result")
        bootstrap_sent = False
        return

    await broadcast(WSMessage(type="scene_analysis", payload={
        "scene_type": analysis.scene_type,
        "scene_description": analysis.scene_description,
        "zones": analysis.zones,
        "suggested_rules": analysis.suggested_rules,
    }))


# ---------------------------------------------------------------------------
# Alert verification + actions
# ---------------------------------------------------------------------------

async def _verify_and_broadcast_alert(alert: Alert, frame: np.ndarray) -> None:
    try:
        result = await narrator.verify(frame, alert)
        if not result.confirmed:
            log.info("Alert '%s' rejected by LLM verification", alert.rule_name)
            return
        alert.narration = result.note
        alerts.append(alert)
        await broadcast(WSMessage(type="alert", payload=alert.model_dump()))
        if result.note:
            await broadcast(WSMessage(
                type="narration",
                payload={"alert_id": alert.id, "narration": result.note},
            ))
        # Block 3: Execute actions
        asyncio.create_task(action_engine.execute(alert, broadcast_event))
    except Exception as e:
        log.error("Verification failed: %s", e)
        # On error, still broadcast (don't suppress real alerts)
        alerts.append(alert)
        await broadcast(WSMessage(type="alert", payload=alert.model_dump()))


# ---------------------------------------------------------------------------
# Block 6: Anomaly alert
# ---------------------------------------------------------------------------

async def _handle_anomaly_alert(frame: np.ndarray, score: float, timestamp: float) -> None:
    """When anomaly score exceeds threshold, compare against baseline."""
    try:
        # Get a baseline frame for comparison
        baseline_frames = anomaly_detector._baseline_frames
        if baseline_frames:
            # Use middle baseline frame as representative (stored in color at 480x360)
            baseline = baseline_frames[len(baseline_frames) // 2]
            description = await narrator.compare_anomaly(baseline, frame, score)
        else:
            description = await narrator.narrate_scene(frame, latest_detections)
        if not description:
            description = "Anomalous activity detected."

        await broadcast(WSMessage(type="anomaly_detected", payload={
            "score": round(score, 3),
            "description": description,
            "frame_b64": frame_to_b64(frame, quality=70),
            "timestamp": timestamp,
        }))

        # Also create a regular alert for the anomaly
        anomaly_alert = Alert(
            rule_id="anomaly",
            rule_name="Anomaly Detection",
            severity="high" if score > 0.6 else "medium",
            timestamp=timestamp,
            frame_b64=frame_to_b64(frame, quality=80),
            narration=description,
            detections=latest_detections[:5],
        )
        alerts.append(anomaly_alert)
        await broadcast(WSMessage(type="alert", payload=anomaly_alert.model_dump()))
        asyncio.create_task(action_engine.execute(anomaly_alert, broadcast_event))

    except Exception as e:
        log.error("Anomaly alert handling failed: %s", e)


# ---------------------------------------------------------------------------
# Block 2: Reasoning loop
# ---------------------------------------------------------------------------

async def reasoning_loop() -> None:
    """Background loop that runs LLM reasoning every ~10 seconds."""
    while camera_running:
        await asyncio.sleep(10)

        if not reasoning_enabled or not connected_clients:
            continue

        # Grab frames from replay buffer (last 10 seconds, evenly spaced)
        now = time.time()
        all_frames = replay_buffer.get_frames(now - 10, 10)
        if not all_frames:
            continue

        # Sample up to 4 frames evenly
        step = max(1, len(all_frames) // 4)
        sampled = all_frames[::step][:4]

        insight = await reasoner.analyze(
            frames=sampled,
            detections=latest_detections,
            active_rules=rules,
            active_zones=zones,
            recent_alerts=alerts[-10:],
        )

        await broadcast(WSMessage(type="insight", payload={
            "observation": insight.observation,
            "concerns": insight.concerns,
            "suggested_alerts": insight.suggested_alerts,
            "prediction": insight.prediction,
            "timestamp": now,
        }))

        # Process any suggested alerts from the reasoning engine (with cooldown)
        last_reasoning_alert = getattr(reasoning_loop, "_last_alert", 0.0)
        if insight.suggested_alerts and now - last_reasoning_alert > 60.0:
            reasoning_loop._last_alert = now  # type: ignore[attr-defined]
            # Only take the first (most important) suggestion per cycle
            sa = insight.suggested_alerts[0]
            alert = Alert(
                rule_id="reasoning",
                rule_name="AI Reasoning",
                severity=sa.get("severity", "medium"),
                timestamp=now,
                frame_b64=frame_to_b64(latest_frame, quality=80) if latest_frame is not None else "",
                narration=sa.get("reason", ""),
                detections=latest_detections[:5],
            )
            alerts.append(alert)
            await broadcast(WSMessage(type="alert", payload=alert.model_dump()))
            asyncio.create_task(action_engine.execute(alert, broadcast_event))


# ---------------------------------------------------------------------------
# Block 4: Memory loop
# ---------------------------------------------------------------------------

async def memory_loop() -> None:
    """Background loop that creates scene memory entries every ~30 seconds."""
    while camera_running:
        await asyncio.sleep(30)

        if latest_frame is None or not connected_clients:
            continue

        now = time.time()
        entry = await scene_memory.add_entry(
            frame=latest_frame,
            detections=latest_detections,
            recent_alerts=alerts[-10:],
            timestamp=now,
        )

        if entry:
            await broadcast(WSMessage(type="memory_entry", payload={
                "timestamp": entry.timestamp,
                "summary": entry.summary,
            }))


# ---------------------------------------------------------------------------
# Block 5: Narration loop
# ---------------------------------------------------------------------------

async def narration_loop() -> None:
    """Background loop for continuous live narration every ~8 seconds."""
    while camera_running:
        await asyncio.sleep(8)

        if not narration_enabled or latest_frame is None or not connected_clients:
            continue

        text = await narrator.narrate_scene(latest_frame, latest_detections)
        if text:
            now = time.time()
            await broadcast(WSMessage(type="live_narration", payload={
                "text": text,
                "timestamp": now,
            }))


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks: list[asyncio.Task] = []
    if os.getenv("WATCHTOWER_NO_CAMERA") != "1":
        tasks.append(asyncio.create_task(camera_loop()))
        tasks.append(asyncio.create_task(reasoning_loop()))
        tasks.append(asyncio.create_task(memory_loop()))
        tasks.append(asyncio.create_task(narration_loop()))
    else:
        log.info("Camera disabled (WATCHTOWER_NO_CAMERA=1)")
    yield
    global camera_running
    camera_running = False
    for task in tasks:
        task.cancel()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="WatchTower", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Remote camera WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/camera")
async def camera_feed_endpoint(ws: WebSocket) -> None:
    """Accepts frames from remote camera clients (Pi, phone, etc).

    Camera clients send binary JPEG frames or JSON with base64 frame.
    """
    global remote_frame, remote_camera_connected
    await ws.accept()
    remote_camera_connected = True
    log.info("Remote camera connected")

    try:
        while True:
            data = await ws.receive()
            if "bytes" in data and data["bytes"]:
                # Binary JPEG frame — decode directly
                buf = np.frombuffer(data["bytes"], dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is not None:
                    remote_frame = frame
            elif "text" in data and data["text"]:
                # JSON with base64 frame
                msg = json.loads(data["text"])
                if "frame" in msg:
                    import base64 as b64mod
                    raw = b64mod.b64decode(msg["frame"])
                    buf = np.frombuffer(raw, dtype=np.uint8)
                    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if frame is not None:
                        remote_frame = frame
    except WebSocketDisconnect:
        pass
    finally:
        remote_camera_connected = False
        remote_frame = None
        log.info("Remote camera disconnected")


# ---------------------------------------------------------------------------
# Frontend WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    connected_clients.append(ws)
    log.info("Client connected (%d total)", len(connected_clients))

    # Send current state on connect
    await ws.send_text(WSMessage(
        type="init",
        payload={
            "zones": [z.model_dump() for z in zones],
            "rules": [r.model_dump() for r in rules],
            "alerts": [a.model_dump() for a in alerts[-50:]],
            "reasoning_enabled": reasoning_enabled,
            "narration_enabled": narration_enabled,
            "anomaly_phase": anomaly_detector.phase.value,
            "action_config": action_engine.config,
            "camera_source": str(camera_source),
        },
    ).model_dump_json())

    try:
        while True:
            raw = await ws.receive_text()
            msg = WSMessage.model_validate_json(raw)
            await _handle_message(ws, msg)
    except WebSocketDisconnect:
        pass
    finally:
        if ws in connected_clients:
            connected_clients.remove(ws)
        log.info("Client disconnected (%d total)", len(connected_clients))


async def _handle_message(ws: WebSocket, msg: WSMessage) -> None:
    handler = _message_handlers.get(msg.type)
    if handler:
        await handler(ws, msg.payload)
    else:
        log.warning("Unknown message type: %s", msg.type)


# ---------------------------------------------------------------------------
# Message handlers
# ---------------------------------------------------------------------------

async def _handle_add_rule(ws: WebSocket, payload: dict[str, Any]) -> None:
    text = payload.get("text", "")
    if not text:
        return

    severity = payload.get("severity", "medium")
    zone_names = [z.name for z in zones]
    result = await rule_parser.parse(text, zone_names, severity=severity)
    if result:
        rule, missing_zones = result
        rules.append(rule)
        rule_payload = rule.model_dump()
        if missing_zones:
            rule_payload["_missing_zones"] = missing_zones
        await broadcast(WSMessage(
            type="rule_added",
            payload=rule_payload,
        ))


async def _handle_update_rule(ws: WebSocket, payload: dict[str, Any]) -> None:
    rule_id = payload.get("id", "")
    for i, r in enumerate(rules):
        if r.id == rule_id:
            updated = r.model_copy(update={
                k: v for k, v in payload.items()
                if k != "id" and hasattr(r, k)
            })
            rules[i] = updated
            await broadcast(WSMessage(
                type="rule_updated",
                payload=updated.model_dump(),
            ))
            return


async def _handle_delete_rule(ws: WebSocket, payload: dict[str, Any]) -> None:
    rule_id = payload.get("id", "")
    for i, r in enumerate(rules):
        if r.id == rule_id:
            rules.pop(i)
            await broadcast(WSMessage(
                type="rule_deleted",
                payload={"id": rule_id},
            ))
            return


async def _handle_toggle_rule(ws: WebSocket, payload: dict[str, Any]) -> None:
    rule_id = payload.get("id", "")
    for i, r in enumerate(rules):
        if r.id == rule_id:
            updated = r.model_copy(update={"enabled": not r.enabled})
            rules[i] = updated
            await broadcast(WSMessage(
                type="rule_updated",
                payload=updated.model_dump(),
            ))
            return


async def _handle_update_zones(ws: WebSocket, payload: dict[str, Any]) -> None:
    global zones
    raw_zones = payload.get("zones", [])
    zones = [Zone.model_validate(z) for z in raw_zones]
    await broadcast(WSMessage(
        type="zones_updated",
        payload={"zones": [z.model_dump() for z in zones]},
    ))


async def _handle_auto_zones(ws: WebSocket, payload: dict[str, Any]) -> None:
    global zones
    if camera is None or not camera.isOpened():
        return

    ok, frame = camera.read()
    if not ok:
        return

    generated = await zone_generator.generate(frame)
    zones = generated
    await broadcast(WSMessage(
        type="zones_updated",
        payload={"zones": [z.model_dump() for z in zones]},
    ))


async def _handle_get_replay(ws: WebSocket, payload: dict[str, Any]) -> None:
    timestamp = payload.get("timestamp", 0.0)
    duration = payload.get("duration", 10.0)
    frames = replay_buffer.get_frames(timestamp, duration)
    await ws.send_text(WSMessage(
        type="replay",
        payload={
            "frames": [
                {"frame": frame_to_b64(f, quality=50), "timestamp": t}
                for f, t in frames
            ],
        },
    ).model_dump_json())


async def _handle_get_replay_timestamps(ws: WebSocket, payload: dict[str, Any]) -> None:
    timestamps = replay_buffer.get_timestamps()
    await ws.send_text(WSMessage(
        type="replay_timestamps",
        payload={
            "start": timestamps[0] if timestamps else 0,
            "end": timestamps[-1] if timestamps else 0,
            "count": len(timestamps),
        },
    ).model_dump_json())


async def _handle_get_frame_at(ws: WebSocket, payload: dict[str, Any]) -> None:
    timestamp = payload.get("timestamp", 0.0)
    result = replay_buffer.get_frame_at(timestamp)
    if result is None:
        await ws.send_text(WSMessage(
            type="replay_frame",
            payload={"frame": None, "timestamp": 0},
        ).model_dump_json())
        return
    frame, ts = result
    await ws.send_text(WSMessage(
        type="replay_frame",
        payload={"frame": frame_to_b64(frame, quality=60), "timestamp": ts},
    ).model_dump_json())


async def _handle_clear_alerts(ws: WebSocket, payload: dict[str, Any]) -> None:
    global alerts
    alerts = []
    await broadcast(WSMessage(type="alerts_cleared", payload={}))


async def _handle_clear_rules(ws: WebSocket, payload: dict[str, Any]) -> None:
    global rules
    rules = []
    rule_engine._last_fired.clear()
    rule_engine._duration_tracking.clear()
    await broadcast(WSMessage(type="rules_cleared", payload={}))


async def _handle_reset_all(ws: WebSocket, payload: dict[str, Any]) -> None:
    """Reset everything back to initial state and re-trigger bootstrap."""
    global zones, rules, alerts, bootstrap_sent, reasoning_enabled, narration_enabled

    # Clear all state
    zones = []
    rules = []
    alerts = []
    pending_plans.clear()
    rule_engine._last_fired.clear()
    rule_engine._duration_tracking.clear()
    scene_memory._entries.clear()
    anomaly_detector.stop()
    reasoning_enabled = False
    narration_enabled = False
    bootstrap_sent = False

    # Broadcast all clears
    await broadcast(WSMessage(type="zones_updated", payload={"zones": []}))
    await broadcast(WSMessage(type="rules_cleared", payload={}))
    await broadcast(WSMessage(type="alerts_cleared", payload={}))
    await broadcast(WSMessage(type="reasoning_toggled", payload={"enabled": False}))
    await broadcast(WSMessage(type="narration_toggled", payload={"enabled": False}))
    await broadcast(WSMessage(type="anomaly_status", payload={"phase": "off", "time_remaining": 0}))

    log.info("Full reset — re-triggering bootstrap")

    # Re-trigger bootstrap from current frame
    if camera is not None and camera.isOpened():
        ok, frame = camera.read()
        if ok:
            asyncio.create_task(_auto_bootstrap(frame))


async def _handle_switch_camera(ws: WebSocket, payload: dict[str, Any]) -> None:
    """Switch camera source at runtime. Accepts index (0, 1) or URL (rtsp://, http://)."""
    global camera, camera_source, bootstrap_sent
    source = payload.get("source", 0)

    # Parse source: integer index or string URL
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    camera_source = source

    # Close current camera
    if camera is not None and camera.isOpened():
        camera.release()

    # Open new source
    camera = cv2.VideoCapture(camera_source)
    if not camera.isOpened():
        log.error("Cannot open camera source: %s", camera_source)
        await ws.send_text(WSMessage(
            type="camera_error",
            payload={"error": f"Cannot open camera: {camera_source}"},
        ).model_dump_json())
        return

    log.info("Switched camera to: %s", camera_source)

    # Re-trigger bootstrap for the new view
    bootstrap_sent = False
    ok, frame = camera.read()
    if ok:
        asyncio.create_task(_auto_bootstrap(frame))

    await broadcast(WSMessage(type="camera_switched", payload={"source": str(camera_source)}))


# --- Plan generator handlers (API repo only) ---

async def _handle_generate_plan(ws: WebSocket, payload: dict[str, Any]) -> None:
    text = payload.get("text", "")
    if not text:
        return

    # Auto-generate zones if none exist and camera is available
    plan_zones: list[Zone] = []
    zone_names = [z.name for z in zones]
    if not zones and camera is not None and camera.isOpened():
        ok, frame = camera.read()
        if ok:
            generated = await zone_generator.generate(frame)
            plan_zones = generated
            zone_names = [z.name for z in generated]

    # Classify and generate
    result = await plan_generator.classify_and_generate(text, zone_names)
    if result is None:
        await ws.send_text(WSMessage(
            type="plan_error",
            payload={"error": "Failed to process. Try again."},
        ).model_dump_json())
        return

    if result["type"] == "rule":
        # Single rule — use existing flow
        rule = result["rule"]
        rules.append(rule)
        rule_payload = rule.model_dump()
        missing = result.get("missing_zones", [])
        if missing:
            rule_payload["_missing_zones"] = missing
        await broadcast(WSMessage(type="rule_added", payload=rule_payload))

    elif result["type"] == "scenario":
        # Multi-rule plan — store pending and send to requesting client
        plan_data = result["plan"]
        plan = MonitoringPlan(
            name=plan_data["name"],
            description=plan_data["description"],
            scenario=plan_data["scenario"],
            rules=plan_data["rules"],
            zones=plan_zones,
        )
        pending_plans[plan.id] = plan
        await ws.send_text(WSMessage(
            type="plan_generated",
            payload=plan.model_dump(),
        ).model_dump_json())


async def _handle_apply_plan(ws: WebSocket, payload: dict[str, Any]) -> None:
    global zones
    plan_id = payload.get("plan_id", "")
    plan = pending_plans.pop(plan_id, None)
    if plan is None:
        return

    # Commit zones if the plan generated new ones
    if plan.zones:
        zones = plan.zones
        await broadcast(WSMessage(
            type="zones_updated",
            payload={"zones": [z.model_dump() for z in zones]},
        ))

    # Commit each rule
    for rule in plan.rules:
        rules.append(rule)
        await broadcast(WSMessage(type="rule_added", payload=rule.model_dump()))

    # Confirm
    await broadcast(WSMessage(
        type="plan_applied",
        payload={"plan_id": plan_id},
    ))


# --- Block 1: Bootstrap handlers ---

async def _handle_approve_bootstrap(ws: WebSocket, payload: dict[str, Any]) -> None:
    global zones
    from models import Condition

    raw_zones = payload.get("zones", [])
    raw_rules = payload.get("rules", [])

    # Create zones
    colors = [
        "#22d3ee", "#a78bfa", "#34d399", "#fb923c",
        "#f472b6", "#facc15", "#60a5fa", "#e879f9",
    ]
    new_zones: list[Zone] = []
    for i, z in enumerate(raw_zones):
        new_zones.append(Zone(
            name=z["name"],
            x=float(z.get("x", 0)),
            y=float(z.get("y", 0)),
            width=float(z.get("width", 0)),
            height=float(z.get("height", 0)),
            color=colors[i % len(colors)],
        ))
    zones = new_zones

    # Create rules
    new_rules: list[Rule] = []
    for r in raw_rules:
        conditions = [
            Condition(type=c["type"], params=c.get("params", {}))
            for c in r.get("conditions", [])
        ]
        new_rules.append(Rule(
            name=r.get("name", "Unnamed"),
            natural_language=r.get("natural_language", r.get("name", "")),
            conditions=conditions,
            severity=r.get("severity", "medium"),
        ))
    rules.extend(new_rules)

    # Broadcast updates
    await broadcast(WSMessage(
        type="zones_updated",
        payload={"zones": [z.model_dump() for z in zones]},
    ))
    for rule in new_rules:
        await broadcast(WSMessage(type="rule_added", payload=rule.model_dump()))

    log.info("Bootstrap approved: %d zones, %d rules", len(new_zones), len(new_rules))


async def _handle_dismiss_bootstrap(ws: WebSocket, payload: dict[str, Any]) -> None:
    log.info("Bootstrap dismissed by user")


# --- Block 2: Reasoning toggle ---

async def _handle_toggle_reasoning(ws: WebSocket, payload: dict[str, Any]) -> None:
    global reasoning_enabled
    reasoning_enabled = not reasoning_enabled
    await broadcast(WSMessage(type="reasoning_toggled", payload={"enabled": reasoning_enabled}))
    log.info("Reasoning %s", "enabled" if reasoning_enabled else "disabled")


# --- Block 3: Action config ---

async def _handle_update_actions(ws: WebSocket, payload: dict[str, Any]) -> None:
    config = payload.get("config", {})
    if config:
        action_engine.update_config(config)
    await broadcast(WSMessage(type="actions_updated", payload={"config": action_engine.config}))


# --- Block 4: Investigation ---

async def _handle_ask(ws: WebSocket, payload: dict[str, Any]) -> None:
    question = payload.get("question", "")
    if not question:
        return

    answer = await scene_memory.investigate(
        question=question,
        recent_alerts=alerts,
    )

    # Try to find relevant frames from replay buffer
    relevant_frames: list[dict] = []
    # For now, include the current frame as context
    if latest_frame is not None:
        relevant_frames.append({
            "frame": frame_to_b64(latest_frame, quality=50),
            "timestamp": time.time(),
        })

    await ws.send_text(WSMessage(type="ask_response", payload={
        "answer": answer,
        "relevant_frames": relevant_frames,
        "question": question,
    }).model_dump_json())


# --- Block 5: Narration toggle ---

async def _handle_toggle_narration(ws: WebSocket, payload: dict[str, Any]) -> None:
    global narration_enabled
    narration_enabled = not narration_enabled
    await broadcast(WSMessage(type="narration_toggled", payload={"enabled": narration_enabled}))
    log.info("Live narration %s", "enabled" if narration_enabled else "disabled")


# --- Block 6: Anomaly handlers ---

async def _handle_toggle_anomaly(ws: WebSocket, payload: dict[str, Any]) -> None:
    if anomaly_detector.phase == AnomalyPhase.OFF:
        anomaly_detector.start_learning()
        await broadcast(WSMessage(type="anomaly_status", payload={
            "phase": "learning",
            "time_remaining": anomaly_detector.learning_time_remaining,
        }))
    else:
        anomaly_detector.stop()
        await broadcast(WSMessage(type="anomaly_status", payload={
            "phase": "off",
            "time_remaining": 0,
        }))


async def _handle_set_anomaly_threshold(ws: WebSocket, payload: dict[str, Any]) -> None:
    threshold = payload.get("threshold", 0.35)
    anomaly_detector.threshold = float(threshold)
    await broadcast(WSMessage(type="anomaly_status", payload={
        "phase": anomaly_detector.phase.value,
        "time_remaining": anomaly_detector.learning_time_remaining,
        "threshold": anomaly_detector.threshold,
    }))


# ---------------------------------------------------------------------------
# Message handler registry
# ---------------------------------------------------------------------------

_message_handlers = {
    "add_rule": _handle_add_rule,
    "update_rule": _handle_update_rule,
    "delete_rule": _handle_delete_rule,
    "toggle_rule": _handle_toggle_rule,
    "update_zones": _handle_update_zones,
    "auto_zones": _handle_auto_zones,
    "get_replay": _handle_get_replay,
    "get_replay_timestamps": _handle_get_replay_timestamps,
    "get_frame_at": _handle_get_frame_at,
    "clear_alerts": _handle_clear_alerts,
    "clear_rules": _handle_clear_rules,
    "reset_all": _handle_reset_all,
    "switch_camera": _handle_switch_camera,
    # Plan generator (API repo)
    "generate_plan": _handle_generate_plan,
    "apply_plan": _handle_apply_plan,
    # Block 1
    "approve_bootstrap": _handle_approve_bootstrap,
    "dismiss_bootstrap": _handle_dismiss_bootstrap,
    # Block 2
    "toggle_reasoning": _handle_toggle_reasoning,
    # Block 3
    "update_actions": _handle_update_actions,
    # Block 4
    "ask": _handle_ask,
    # Block 5
    "toggle_narration": _handle_toggle_narration,
    # Block 6
    "toggle_anomaly": _handle_toggle_anomaly,
    "set_anomaly_threshold": _handle_set_anomaly_threshold,
}
