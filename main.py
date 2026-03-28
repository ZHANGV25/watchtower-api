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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import database as db
from actions import ActionEngine
from anomaly import AnomalyPhase
from camera_manager import CameraManager, CameraSession
from detector import Detector
from middleware import verify_ws_token
from models import Alert, Camera, Condition, MonitoringPlan, Rule, WSMessage, Zone
from narrator import Narrator
from plan_generator import PlanGenerator
from reasoner import Reasoner
from rule_parser import RuleParser
from scene_analyzer import SceneAnalyzer
from storage import create_frame_store
from zone_generator import ZoneGenerator

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("watchtower")

# ---------------------------------------------------------------------------
# Shared singletons (one instance, shared across all cameras)
# ---------------------------------------------------------------------------

detector = Detector()
narrator = Narrator()
reasoner = Reasoner()
rule_parser = RuleParser()
plan_generator = PlanGenerator()
zone_generator = ZoneGenerator()
scene_analyzer = SceneAnalyzer()
action_engine = ActionEngine()
frame_store = create_frame_store()
camera_mgr = CameraManager()

# Pending plans (keyed by plan ID, temporary)
pending_plans: dict[str, MonitoringPlan] = {}

# Frontend client tracking: ws -> subscribed camera_id
frontend_clients: dict[WebSocket, str] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def frame_to_b64(frame: np.ndarray, quality: int = 70) -> str:
    if frame is None or frame.size == 0:
        return ""
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


async def broadcast(session: CameraSession, msg: WSMessage) -> None:
    """Send message to all frontend viewers of this camera."""
    raw = msg.model_dump_json()
    dead: list[WebSocket] = []
    for ws in session.viewers:
        try:
            await ws.send_text(raw)
        except Exception:
            dead.append(ws)
    for ws in dead:
        session.viewers.remove(ws)
        frontend_clients.pop(ws, None)


async def broadcast_event(session: CameraSession, event_type: str, payload: dict[str, Any]) -> None:
    """Helper for action engine callbacks (wraps broadcast)."""
    await broadcast(session, WSMessage(type=event_type, payload=payload))


def _persist(coro) -> None:
    """Fire-and-forget a DB write coroutine."""
    asyncio.create_task(coro)


# ---------------------------------------------------------------------------
# Per-camera processing loops
# ---------------------------------------------------------------------------

async def camera_processing_loop(session: CameraSession) -> None:
    """Main detection + rule evaluation loop for one camera."""
    log.info("Processing started for camera %s", session.camera_id)
    frame_interval = 1.0 / 24
    last_frame_time = 0.0
    first_frame = True

    while True:
        now = time.time()
        if now - last_frame_time < frame_interval:
            await asyncio.sleep(0.003)
            continue

        frame = session.latest_frame
        if frame is None:
            await asyncio.sleep(0.05)
            continue

        last_frame_time = now

        # Auto-bootstrap on first frame
        if first_frame and not session.bootstrap_sent:
            first_frame = False
            session.bootstrap_sent = True
            asyncio.create_task(_auto_bootstrap(session, frame))

        # YOLO + pose detection
        pose_types = {"person_pose", "person_falling"}
        need_pose = any(
            c.type in pose_types for r in session.rules if r.enabled for c in r.conditions
        )
        loop = asyncio.get_event_loop()
        detections = await loop.run_in_executor(None, detector.detect, frame, need_pose)
        session.latest_detections = detections

        # Replay buffer
        session.replay_buffer.add_frame(frame, now)

        # Anomaly detection
        ad = session.anomaly_detector
        if ad.phase == AnomalyPhase.LEARNING:
            done = ad.learn_frame(frame)
            if done:
                await broadcast(session, WSMessage(type="anomaly_status", payload={"phase": "detecting", "time_remaining": 0}))
        elif ad.phase == AnomalyPhase.DETECTING:
            last_check = getattr(session, "_last_anomaly_check", 0.0)
            if now - last_check > 2.0:
                session._last_anomaly_check = now  # type: ignore[attr-defined]
                score = ad.detect(frame)
                await broadcast(session, WSMessage(type="anomaly_score", payload={"score": round(score, 3), "timestamp": now}))
                last_alert_t = getattr(session, "_last_anomaly_alert", 0.0)
                if score > ad.threshold and now - last_alert_t > 30.0:
                    session._last_anomaly_alert = now  # type: ignore[attr-defined]
                    asyncio.create_task(_handle_anomaly_alert(session, frame, score, now))

        # Rule evaluation
        fired = session.rule_engine.evaluate(session.rules, session.zones, detections, now)
        for alert in fired:
            alert.camera_id = session.camera_id
            alert.frame_b64 = frame_to_b64(frame, quality=80)
            asyncio.create_task(_verify_and_broadcast_alert(session, alert, frame))

        # Broadcast frame
        frame_b64 = await loop.run_in_executor(None, frame_to_b64, frame, 50)
        await broadcast(session, WSMessage(type="frame", payload={
            "frame": frame_b64,
            "detections": [d.model_dump() for d in detections],
            "timestamp": now,
            "fps": round(1.0 / max(time.time() - now, 0.001)),
        }))

        # Heartbeat to DB every 30s
        last_hb = getattr(session, "_last_heartbeat", 0.0)
        if now - last_hb > 30.0:
            session._last_heartbeat = now  # type: ignore[attr-defined]
            _persist(db.camera_heartbeat(session.camera_id))


async def reasoning_loop(session: CameraSession) -> None:
    while True:
        await asyncio.sleep(10)
        if not session.reasoning_enabled or not session.viewers:
            continue
        now = time.time()
        all_frames = session.replay_buffer.get_frames(now - 10, 10)
        if not all_frames:
            continue
        step = max(1, len(all_frames) // 4)
        sampled = all_frames[::step][:4]
        insight = await reasoner.analyze(sampled, session.latest_detections, session.rules, session.zones, session.alerts[-10:])
        await broadcast(session, WSMessage(type="insight", payload={
            "observation": insight.observation, "concerns": insight.concerns,
            "suggested_alerts": insight.suggested_alerts, "prediction": insight.prediction, "timestamp": now,
        }))
        last_ra = getattr(session, "_last_reasoning_alert", 0.0)
        if insight.suggested_alerts and now - last_ra > 60.0:
            session._last_reasoning_alert = now  # type: ignore[attr-defined]
            sa = insight.suggested_alerts[0]
            alert = Alert(camera_id=session.camera_id, rule_id="reasoning", rule_name="AI Reasoning",
                          severity=sa.get("severity", "medium"), timestamp=now,
                          frame_b64=frame_to_b64(session.latest_frame, quality=80) if session.latest_frame is not None else "",
                          narration=sa.get("reason", ""), detections=session.latest_detections[:5])
            session.alerts.append(alert)
            await broadcast(session, WSMessage(type="alert", payload=alert.model_dump()))
            _persist(db.create_alert(alert))
            asyncio.create_task(action_engine.execute(alert, lambda et, p: broadcast_event(session, et, p)))


async def memory_loop(session: CameraSession) -> None:
    while True:
        await asyncio.sleep(30)
        if session.latest_frame is None or not session.viewers:
            continue
        now = time.time()
        entry = await session.scene_memory.add_entry(session.latest_frame, session.latest_detections, session.alerts[-10:], now)
        if entry:
            entry.camera_id = session.camera_id
            _persist(db.create_memory_entry(session.camera_id, entry))


async def narration_loop(session: CameraSession) -> None:
    while True:
        await asyncio.sleep(8)
        if not session.narration_enabled or session.latest_frame is None or not session.viewers:
            continue
        text = await narrator.narrate_scene(session.latest_frame, session.latest_detections)
        if text:
            await broadcast(session, WSMessage(type="live_narration", payload={"text": text, "timestamp": time.time()}))


# ---------------------------------------------------------------------------
# Alert handling
# ---------------------------------------------------------------------------

async def _verify_and_broadcast_alert(session: CameraSession, alert: Alert, frame: np.ndarray) -> None:
    try:
        result = await narrator.verify(frame, alert)
        if not result.confirmed:
            return
        alert.narration = result.note
        session.alerts.append(alert)
        if len(session.alerts) > 100:
            session.alerts = session.alerts[-100:]
        await broadcast(session, WSMessage(type="alert", payload=alert.model_dump()))
        if result.note:
            await broadcast(session, WSMessage(type="narration", payload={"alert_id": alert.id, "narration": result.note}))
        # Persist frame + alert
        frame_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
        path = await frame_store.save_frame(alert.id, frame_bytes)
        alert.frame_path = path
        _persist(db.create_alert(alert, frame_path=path))
        asyncio.create_task(action_engine.execute(alert, lambda et, p: broadcast_event(session, et, p)))
    except Exception as e:
        log.error("Verification failed: %s", e)
        session.alerts.append(alert)
        await broadcast(session, WSMessage(type="alert", payload=alert.model_dump()))


async def _handle_anomaly_alert(session: CameraSession, frame: np.ndarray, score: float, timestamp: float) -> None:
    try:
        baseline_frames = session.anomaly_detector._baseline_frames
        if baseline_frames:
            baseline = baseline_frames[len(baseline_frames) // 2]
            description = await narrator.compare_anomaly(baseline, frame, score)
        else:
            description = await narrator.narrate_scene(frame, session.latest_detections)
        if not description:
            description = "Anomalous activity detected."
        alert = Alert(camera_id=session.camera_id, rule_id="anomaly", rule_name="Anomaly Detection",
                      severity="high" if score > 0.6 else "medium", timestamp=timestamp,
                      frame_b64=frame_to_b64(frame, quality=80), narration=description, detections=session.latest_detections[:5])
        session.alerts.append(alert)
        await broadcast(session, WSMessage(type="alert", payload=alert.model_dump()))
        await broadcast(session, WSMessage(type="anomaly_detected", payload={
            "score": round(score, 3), "description": description, "frame_b64": alert.frame_b64, "timestamp": timestamp}))
        _persist(db.create_alert(alert))
        asyncio.create_task(action_engine.execute(alert, lambda et, p: broadcast_event(session, et, p)))
    except Exception as e:
        log.error("Anomaly alert failed: %s", e)


async def _auto_bootstrap(session: CameraSession, frame: np.ndarray) -> None:
    log.info("Running auto-bootstrap for camera %s", session.camera_id)
    analysis = await scene_analyzer.analyze(frame)
    if not analysis.scene_description:
        session.bootstrap_sent = False
        return
    await broadcast(session, WSMessage(type="scene_analysis", payload={
        "scene_type": analysis.scene_type, "scene_description": analysis.scene_description,
        "zones": analysis.zones, "suggested_rules": analysis.suggested_rules}))


# ---------------------------------------------------------------------------
# Start/stop per-camera tasks
# ---------------------------------------------------------------------------

async def start_camera_tasks(session: CameraSession) -> None:
    """Spawn all background tasks for a camera session."""
    session.processing_task = asyncio.create_task(camera_processing_loop(session))
    asyncio.create_task(reasoning_loop(session))
    asyncio.create_task(memory_loop(session))
    asyncio.create_task(narration_loop(session))


async def load_session_from_db(session: CameraSession) -> None:
    """Load persisted zones, rules, alerts into a session."""
    session.zones = await db.list_zones(session.camera_id)
    session.rules = await db.list_rules(session.camera_id)
    # Load recent alerts (last 50)
    raw_alerts = await db.list_alerts(session.camera_id, limit=50)
    for ra in raw_alerts:
        session.alerts.append(Alert(
            id=ra["id"], camera_id=ra["camera_id"], rule_id=ra["rule_id"],
            rule_name=ra["rule_name"], severity=ra["severity"], timestamp=ra["timestamp"],
            narration=ra.get("narration", ""), frame_path=ra.get("frame_path", ""),
        ))


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()

    # Load all cameras from DB
    cameras = await db.list_cameras()
    if not cameras:
        default_cam = Camera(name="Default Camera")
        await db.create_camera(default_cam)
        cameras = [default_cam]

    for cam in cameras:
        session = camera_mgr.get_or_create_session(cam.id, cam.name)
        await load_session_from_db(session)

    # Optional local camera for backwards compat
    if os.getenv("WATCHTOWER_NO_CAMERA") != "1":
        source = os.getenv("WATCHTOWER_CAMERA", "0")
        cam_source: int | str = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(cam_source)
        if cap.isOpened():
            default_session = camera_mgr.get_session(cameras[0].id)
            if default_session:
                log.info("Local camera %s attached to %s", cam_source, cameras[0].name)
                asyncio.create_task(_local_camera_feeder(cap, default_session))
                await start_camera_tasks(default_session)
        else:
            log.warning("Cannot open local camera %s", cam_source)
            cap.release()

    yield

    # Cleanup
    for session in camera_mgr.list_sessions():
        if session.processing_task:
            session.processing_task.cancel()
    await db.close_db()


async def _local_camera_feeder(cap: cv2.VideoCapture, session: CameraSession) -> None:
    """Feed frames from a local OpenCV camera into a session."""
    frame_interval = 1.0 / 24
    last = 0.0
    while True:
        now = time.time()
        if now - last < frame_interval:
            await asyncio.sleep(0.003)
            continue
        ok, frame = cap.read()
        if not ok:
            await asyncio.sleep(0.01)
            continue
        last = now
        session.latest_frame = frame


# ---------------------------------------------------------------------------
# App + REST routes
# ---------------------------------------------------------------------------

app = FastAPI(title="WatchTower", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Mount REST routes
from routes.auth_routes import router as auth_router
from routes.cameras import router as cameras_router
from routes.zones import router as zones_router
from routes.rules import router as rules_router
from routes.alerts import router as alerts_router
from routes.clips import router as clips_router, init_clip_processor

app.include_router(auth_router)
app.include_router(cameras_router)
app.include_router(zones_router)
app.include_router(rules_router)
app.include_router(alerts_router)
app.include_router(clips_router)

# Initialize clip processor with shared singletons
init_clip_processor(detector, narrator, action_engine, frame_store, camera_mgr)

# Serve stored frames
os.makedirs("data/frames", exist_ok=True)
app.mount("/data", StaticFiles(directory="data"), name="data")


# ---------------------------------------------------------------------------
# Camera WebSocket (devices push frames here)
# ---------------------------------------------------------------------------

@app.websocket("/ws/camera/{camera_id}")
async def camera_feed_endpoint(ws: WebSocket, camera_id: str) -> None:
    await ws.accept()

    # Auto-register camera if new
    cam = await db.get_camera(camera_id)
    if not cam:
        cam = Camera(id=camera_id, name=f"Camera {camera_id[:4]}")
        await db.create_camera(cam)

    await db.camera_heartbeat(camera_id)
    session = camera_mgr.get_or_create_session(camera_id, cam.name)
    session.camera_ws = ws
    await load_session_from_db(session)
    await start_camera_tasks(session)
    log.info("Camera %s connected", camera_id)

    # Notify frontend clients
    for fws in frontend_clients:
        try:
            await fws.send_text(WSMessage(type="camera_online", payload={"camera_id": camera_id, "name": cam.name}).model_dump_json())
        except Exception:
            pass

    try:
        while True:
            data = await ws.receive()
            if "bytes" in data and data["bytes"]:
                buf = np.frombuffer(data["bytes"], dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is not None:
                    session.latest_frame = frame
            elif "text" in data and data["text"]:
                msg = json.loads(data["text"])
                if "frame" in msg:
                    raw = base64.b64decode(msg["frame"])
                    buf = np.frombuffer(raw, dtype=np.uint8)
                    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if frame is not None:
                        session.latest_frame = frame
    except WebSocketDisconnect:
        pass
    finally:
        session.camera_ws = None
        session.latest_frame = None
        if session.processing_task:
            session.processing_task.cancel()
            session.processing_task = None
        await db.camera_offline(camera_id)
        log.info("Camera %s disconnected", camera_id)
        for fws in frontend_clients:
            try:
                await fws.send_text(WSMessage(type="camera_offline", payload={"camera_id": camera_id}).model_dump_json())
            except Exception:
                pass


# Backwards compat: /ws/camera without ID
@app.websocket("/ws/camera")
async def camera_feed_legacy(ws: WebSocket) -> None:
    await ws.accept()
    cameras = await db.list_cameras()
    camera_id = cameras[0].id if cameras else "default"
    session = camera_mgr.get_or_create_session(camera_id)
    session.camera_ws = ws
    if not session.processing_task:
        await start_camera_tasks(session)
    try:
        while True:
            data = await ws.receive()
            if "bytes" in data and data["bytes"]:
                buf = np.frombuffer(data["bytes"], dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is not None:
                    session.latest_frame = frame
    except WebSocketDisconnect:
        pass
    finally:
        session.camera_ws = None


# ---------------------------------------------------------------------------
# Frontend WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, token: str = Query(default=None)) -> None:
    await ws.accept()

    # Optional auth (don't block if no token — dev mode)
    user = verify_ws_token(token) if token else None

    # Default subscribe to first camera
    cameras = await db.list_cameras()
    default_cam_id = cameras[0].id if cameras else "default"
    session = camera_mgr.get_or_create_session(default_cam_id)
    session.viewers.append(ws)
    frontend_clients[ws] = default_cam_id

    log.info("Frontend connected (camera: %s, auth: %s)", default_cam_id, "yes" if user else "no")

    # Send init
    await ws.send_text(WSMessage(type="init", payload={
        "cameras": [c.model_dump() for c in cameras],
        "camera_id": default_cam_id,
        "zones": [z.model_dump() for z in session.zones],
        "rules": [r.model_dump() for r in session.rules],
        "alerts": [a.model_dump() for a in session.alerts[-50:]],
        "reasoning_enabled": session.reasoning_enabled,
        "narration_enabled": session.narration_enabled,
        "anomaly_phase": session.anomaly_detector.phase.value,
        "action_config": action_engine.config,
    }).model_dump_json())

    try:
        while True:
            raw = await ws.receive_text()
            msg = WSMessage.model_validate_json(raw)
            cam_id = frontend_clients.get(ws, default_cam_id)
            sess = camera_mgr.get_session(cam_id) or session
            await _handle_message(sess, ws, msg)
    except WebSocketDisconnect:
        pass
    finally:
        await camera_mgr.remove_viewer(ws)
        frontend_clients.pop(ws, None)
        log.info("Frontend disconnected")


# ---------------------------------------------------------------------------
# Message handlers (operate on per-camera session)
# ---------------------------------------------------------------------------

async def _handle_message(session: CameraSession, ws: WebSocket, msg: WSMessage) -> None:
    handler = _message_handlers.get(msg.type)
    if handler:
        await handler(session, ws, msg.payload)
    else:
        log.warning("Unknown message type: %s", msg.type)


async def _handle_subscribe(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    """Switch which camera this frontend is viewing."""
    camera_id = payload.get("camera_id", "")
    new_session = camera_mgr.get_session(camera_id)
    if not new_session:
        return
    # Remove from old session
    if ws in session.viewers:
        session.viewers.remove(ws)
    # Add to new session
    new_session.viewers.append(ws)
    frontend_clients[ws] = camera_id
    # Send new session state
    await ws.send_text(WSMessage(type="init", payload={
        "camera_id": camera_id,
        "zones": [z.model_dump() for z in new_session.zones],
        "rules": [r.model_dump() for r in new_session.rules],
        "alerts": [a.model_dump() for a in new_session.alerts[-50:]],
        "reasoning_enabled": new_session.reasoning_enabled,
        "narration_enabled": new_session.narration_enabled,
        "anomaly_phase": new_session.anomaly_detector.phase.value,
        "action_config": action_engine.config,
    }).model_dump_json())


async def _handle_add_rule(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    text = payload.get("text", "")
    if not text:
        return
    severity = payload.get("severity", "medium")
    zone_names = [z.name for z in session.zones]
    result = await rule_parser.parse(text, zone_names, severity=severity)
    if result:
        rule, missing = result
        rule.camera_id = session.camera_id
        session.rules.append(rule)
        _persist(db.create_rule(rule))
        rule_payload = rule.model_dump()
        if missing:
            rule_payload["_missing_zones"] = missing
        await broadcast(session, WSMessage(type="rule_added", payload=rule_payload))


async def _handle_update_rule(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    rule_id = payload.get("id", "")
    for i, r in enumerate(session.rules):
        if r.id == rule_id:
            updated = r.model_copy(update={k: v for k, v in payload.items() if k != "id" and hasattr(r, k)})
            session.rules[i] = updated
            _persist(db.update_rule(rule_id, **{k: v for k, v in payload.items() if k != "id" and hasattr(r, k)}))
            await broadcast(session, WSMessage(type="rule_updated", payload=updated.model_dump()))
            return


async def _handle_delete_rule(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    rule_id = payload.get("id", "")
    session.rules = [r for r in session.rules if r.id != rule_id]
    _persist(db.delete_rule(rule_id))
    await broadcast(session, WSMessage(type="rule_deleted", payload={"id": rule_id}))


async def _handle_toggle_rule(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    rule_id = payload.get("id", "")
    for i, r in enumerate(session.rules):
        if r.id == rule_id:
            updated = r.model_copy(update={"enabled": not r.enabled})
            session.rules[i] = updated
            _persist(db.update_rule(rule_id, enabled=not r.enabled))
            await broadcast(session, WSMessage(type="rule_updated", payload=updated.model_dump()))
            return


async def _handle_update_zones(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    raw = payload.get("zones", [])
    session.zones = [Zone.model_validate(z) for z in raw]
    for z in session.zones:
        z.camera_id = session.camera_id
    _persist(db.replace_zones(session.camera_id, session.zones))
    await broadcast(session, WSMessage(type="zones_updated", payload={"zones": [z.model_dump() for z in session.zones]}))


async def _handle_auto_zones(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    if session.latest_frame is None:
        return
    generated = await zone_generator.generate(session.latest_frame)
    for z in generated:
        z.camera_id = session.camera_id
    session.zones = generated
    _persist(db.replace_zones(session.camera_id, generated))
    await broadcast(session, WSMessage(type="zones_updated", payload={"zones": [z.model_dump() for z in generated]}))


async def _handle_get_replay(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    ts = payload.get("timestamp", 0.0)
    dur = payload.get("duration", 10.0)
    frames = session.replay_buffer.get_frames(ts, dur)
    await ws.send_text(WSMessage(type="replay", payload={
        "frames": [{"frame": frame_to_b64(f, quality=50), "timestamp": t} for f, t in frames]
    }).model_dump_json())


async def _handle_get_replay_timestamps(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    ts = session.replay_buffer.get_timestamps()
    await ws.send_text(WSMessage(type="replay_timestamps", payload={
        "start": ts[0] if ts else 0, "end": ts[-1] if ts else 0, "count": len(ts)
    }).model_dump_json())


async def _handle_get_frame_at(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    result = session.replay_buffer.get_frame_at(payload.get("timestamp", 0.0))
    if result is None:
        await ws.send_text(WSMessage(type="replay_frame", payload={"frame": None, "timestamp": 0}).model_dump_json())
        return
    f, t = result
    await ws.send_text(WSMessage(type="replay_frame", payload={"frame": frame_to_b64(f, quality=60), "timestamp": t}).model_dump_json())


async def _handle_clear_alerts(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    session.alerts.clear()
    _persist(db.delete_alerts_for_camera(session.camera_id))
    await broadcast(session, WSMessage(type="alerts_cleared", payload={}))


async def _handle_clear_rules(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    session.rules.clear()
    session.rule_engine._last_fired.clear()
    session.rule_engine._duration_tracking.clear()
    _persist(db.delete_rules_for_camera(session.camera_id))
    await broadcast(session, WSMessage(type="rules_cleared", payload={}))


async def _handle_reset_all(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    session.zones.clear()
    session.rules.clear()
    session.alerts.clear()
    session.rule_engine._last_fired.clear()
    session.rule_engine._duration_tracking.clear()
    session.scene_memory._entries.clear()
    session.anomaly_detector.stop()
    session.reasoning_enabled = False
    session.narration_enabled = False
    session.bootstrap_sent = False
    _persist(db.replace_zones(session.camera_id, []))
    _persist(db.delete_rules_for_camera(session.camera_id))
    _persist(db.delete_alerts_for_camera(session.camera_id))
    await broadcast(session, WSMessage(type="zones_updated", payload={"zones": []}))
    await broadcast(session, WSMessage(type="rules_cleared", payload={}))
    await broadcast(session, WSMessage(type="alerts_cleared", payload={}))
    await broadcast(session, WSMessage(type="reasoning_toggled", payload={"enabled": False}))
    await broadcast(session, WSMessage(type="narration_toggled", payload={"enabled": False}))
    await broadcast(session, WSMessage(type="anomaly_status", payload={"phase": "off", "time_remaining": 0}))
    if session.latest_frame is not None:
        asyncio.create_task(_auto_bootstrap(session, session.latest_frame))


async def _handle_generate_plan(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    text = payload.get("text", "")
    if not text:
        return
    plan_zones: list[Zone] = []
    zone_names = [z.name for z in session.zones]
    if not session.zones and session.latest_frame is not None:
        generated = await zone_generator.generate(session.latest_frame)
        plan_zones = generated
        zone_names = [z.name for z in generated]
    result = await plan_generator.classify_and_generate(text, zone_names)
    if result is None:
        await ws.send_text(WSMessage(type="plan_error", payload={"error": "Failed to process."}).model_dump_json())
        return
    if result["type"] == "rule":
        rule = result["rule"]
        rule.camera_id = session.camera_id
        session.rules.append(rule)
        _persist(db.create_rule(rule))
        rp = rule.model_dump()
        missing = result.get("missing_zones", [])
        if missing:
            rp["_missing_zones"] = missing
        await broadcast(session, WSMessage(type="rule_added", payload=rp))
    elif result["type"] == "scenario":
        pd = result["plan"]
        plan = MonitoringPlan(name=pd["name"], description=pd["description"], scenario=pd["scenario"], rules=pd["rules"], zones=plan_zones)
        pending_plans[plan.id] = plan
        await ws.send_text(WSMessage(type="plan_generated", payload=plan.model_dump()).model_dump_json())


async def _handle_apply_plan(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    plan = pending_plans.pop(payload.get("plan_id", ""), None)
    if not plan:
        return
    if plan.zones:
        for z in plan.zones:
            z.camera_id = session.camera_id
        session.zones = plan.zones
        _persist(db.replace_zones(session.camera_id, plan.zones))
        await broadcast(session, WSMessage(type="zones_updated", payload={"zones": [z.model_dump() for z in plan.zones]}))
    for rule in plan.rules:
        rule.camera_id = session.camera_id
        session.rules.append(rule)
        _persist(db.create_rule(rule))
        await broadcast(session, WSMessage(type="rule_added", payload=rule.model_dump()))
    await broadcast(session, WSMessage(type="plan_applied", payload={"plan_id": plan.id}))


async def _handle_approve_bootstrap(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    colors = ["#22d3ee", "#a78bfa", "#34d399", "#fb923c", "#f472b6", "#facc15", "#60a5fa", "#e879f9"]
    new_zones = [Zone(camera_id=session.camera_id, name=z["name"], x=float(z.get("x", 0)), y=float(z.get("y", 0)),
                      width=float(z.get("width", 0)), height=float(z.get("height", 0)), color=colors[i % len(colors)])
                 for i, z in enumerate(payload.get("zones", []))]
    session.zones = new_zones
    _persist(db.replace_zones(session.camera_id, new_zones))
    new_rules = [Rule(camera_id=session.camera_id, name=r.get("name", ""), natural_language=r.get("natural_language", r.get("name", "")),
                      conditions=[Condition(type=c["type"], params=c.get("params", {})) for c in r.get("conditions", [])],
                      severity=r.get("severity", "medium"))
                 for r in payload.get("rules", [])]
    session.rules.extend(new_rules)
    for rule in new_rules:
        _persist(db.create_rule(rule))
    await broadcast(session, WSMessage(type="zones_updated", payload={"zones": [z.model_dump() for z in new_zones]}))
    for rule in new_rules:
        await broadcast(session, WSMessage(type="rule_added", payload=rule.model_dump()))


async def _handle_dismiss_bootstrap(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    pass


async def _handle_toggle_reasoning(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    session.reasoning_enabled = not session.reasoning_enabled
    await broadcast(session, WSMessage(type="reasoning_toggled", payload={"enabled": session.reasoning_enabled}))


async def _handle_update_actions(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    config = payload.get("config", {})
    if config:
        action_engine.update_config(config)
    await broadcast(session, WSMessage(type="actions_updated", payload={"config": action_engine.config}))


async def _handle_ask(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    question = payload.get("question", "")
    if not question:
        return
    answer = await session.scene_memory.investigate(question=question, recent_alerts=session.alerts)
    relevant_frames: list[dict] = []
    if session.latest_frame is not None:
        relevant_frames.append({"frame": frame_to_b64(session.latest_frame, quality=50), "timestamp": time.time()})
    await ws.send_text(WSMessage(type="ask_response", payload={"answer": answer, "relevant_frames": relevant_frames, "question": question}).model_dump_json())


async def _handle_toggle_narration(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    session.narration_enabled = not session.narration_enabled
    await broadcast(session, WSMessage(type="narration_toggled", payload={"enabled": session.narration_enabled}))


async def _handle_toggle_anomaly(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    if session.anomaly_detector.phase == AnomalyPhase.OFF:
        session.anomaly_detector.start_learning()
        await broadcast(session, WSMessage(type="anomaly_status", payload={"phase": "learning", "time_remaining": session.anomaly_detector.learning_time_remaining}))
    else:
        session.anomaly_detector.stop()
        await broadcast(session, WSMessage(type="anomaly_status", payload={"phase": "off", "time_remaining": 0}))


async def _handle_set_anomaly_threshold(session: CameraSession, ws: WebSocket, payload: dict[str, Any]) -> None:
    session.anomaly_detector.threshold = float(payload.get("threshold", 0.35))
    await broadcast(session, WSMessage(type="anomaly_status", payload={
        "phase": session.anomaly_detector.phase.value, "time_remaining": session.anomaly_detector.learning_time_remaining,
        "threshold": session.anomaly_detector.threshold}))


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

_message_handlers: dict[str, Any] = {
    "subscribe": _handle_subscribe,
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
    "generate_plan": _handle_generate_plan,
    "apply_plan": _handle_apply_plan,
    "approve_bootstrap": _handle_approve_bootstrap,
    "dismiss_bootstrap": _handle_dismiss_bootstrap,
    "toggle_reasoning": _handle_toggle_reasoning,
    "update_actions": _handle_update_actions,
    "ask": _handle_ask,
    "toggle_narration": _handle_toggle_narration,
    "toggle_anomaly": _handle_toggle_anomaly,
    "set_anomaly_threshold": _handle_set_anomaly_threshold,
}
