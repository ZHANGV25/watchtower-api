"""Clip processing REST endpoints.

Cameras upload clips (or notify of S3 uploads). The API processes
the clip through YOLO + rules + LLM and creates alerts.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, UploadFile

import db
from models import Alert

log = logging.getLogger("watchtower.clips")

router = APIRouter(prefix="/api/clips", tags=["clips"])

# These get set by main.py at startup
_detector = None
_rule_engine_factory = None
_narrator = None
_action_engine = None
_frame_store = None
_camera_mgr = None


def init_clip_processor(detector, narrator, action_engine, frame_store, camera_mgr):
    """Called by main.py to inject shared singletons."""
    global _detector, _narrator, _action_engine, _frame_store, _camera_mgr
    _detector = detector
    _narrator = narrator
    _action_engine = action_engine
    _frame_store = frame_store
    _camera_mgr = camera_mgr


@router.post("/process")
async def process_clip_from_s3(body: dict):
    """Process a clip that was uploaded to S3."""
    clip_id = body.get("clip_id", "")
    camera_id = body.get("camera_id", "")
    s3_key = body.get("s3_key", "")
    timestamp = body.get("timestamp", 0.0)

    if not s3_key or not camera_id:
        return {"error": "Missing s3_key or camera_id"}

    # Download clip from S3
    try:
        import boto3
        s3 = boto3.client("s3")
        bucket = os.getenv("WATCHTOWER_S3_BUCKET", "watchtower-clips-008524")
        local_path = os.path.join(tempfile.gettempdir(), f"wt_{clip_id}.mp4")
        s3.download_file(bucket, s3_key, local_path)
    except Exception as e:
        log.error("Failed to download clip from S3: %s", e)
        return {"error": f"S3 download failed: {e}"}

    # Process the clip
    result = await _process_clip_file(local_path, camera_id, timestamp)

    # Clean up
    try:
        os.remove(local_path)
    except OSError:
        pass

    return result


@router.post("/upload")
async def upload_and_process_clip(
    clip: UploadFile = File(...),
    camera_id: str = Form(...),
    timestamp: str = Form("0"),
):
    """Upload a clip directly and process it."""
    # Save to temp file
    local_path = os.path.join(tempfile.gettempdir(), f"wt_{clip.filename}")
    with open(local_path, "wb") as f:
        content = await clip.read()
        f.write(content)

    result = await _process_clip_file(local_path, camera_id, float(timestamp))

    try:
        os.remove(local_path)
    except OSError:
        pass

    return result


async def _process_clip_file(clip_path: str, camera_id: str, timestamp: float) -> dict:
    """Process a video clip: extract frames, run YOLO, evaluate rules, generate alerts."""
    if _detector is None:
        return {"error": "Clip processor not initialized"}

    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        return {"error": f"Cannot open clip: {clip_path}"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_fps = cap.get(cv2.CAP_PROP_FPS) or 15

    log.info("Processing clip for camera %s: %d frames at %.0f fps", camera_id, total_frames, clip_fps)

    # Get camera's rules and zones from DB
    rules = await db.list_rules(camera_id)
    zones = await db.list_zones(camera_id)

    if not rules:
        cap.release()
        return {"status": "no_rules", "message": "No rules configured for this camera"}

    # Import RuleEngine for this clip
    from rule_engine import RuleEngine
    clip_rule_engine = RuleEngine()

    alerts_created: list[dict] = []
    frame_idx = 0
    sample_interval = max(1, int(clip_fps / 5))  # Process ~5 frames per second

    all_detections = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1

        # Sample frames (don't process every single one)
        if frame_idx % sample_interval != 0:
            continue

        frame_time = timestamp + (frame_idx / clip_fps)

        # Run YOLO
        loop = asyncio.get_event_loop()
        detections = await loop.run_in_executor(None, _detector.detect, frame, False)
        all_detections = detections

        # Evaluate rules
        fired = clip_rule_engine.evaluate(rules, zones, detections, frame_time)

        for alert_data in fired:
            alert = Alert(
                camera_id=camera_id,
                rule_id=alert_data.rule_id,
                rule_name=alert_data.rule_name,
                severity=alert_data.severity,
                timestamp=frame_time,
                detections=detections[:5],
            )

            # Verify with LLM
            if _narrator:
                result = await _narrator.verify(frame, alert)
                if not result.confirmed:
                    continue
                alert.narration = result.note

            # Save frame
            if _frame_store:
                frame_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                path = await _frame_store.save_frame(alert.id, frame_bytes)
                alert.frame_path = path

            # Persist alert
            await db.create_alert(alert, frame_path=alert.frame_path)
            alerts_created.append({"id": alert.id, "rule_name": alert.rule_name, "severity": alert.severity})

            # Push to live viewers if any
            if _camera_mgr:
                session = _camera_mgr.get_session(camera_id)
                if session:
                    session.alerts.append(alert)
                    from models import WSMessage
                    for ws in session.viewers:
                        try:
                            await ws.send_text(WSMessage(type="alert", payload=alert.model_dump()).model_dump_json())
                        except Exception:
                            pass

            # Execute actions (push notifications, etc.)
            if _action_engine:
                async def _broadcast(et, p):
                    pass  # No live broadcast needed for clip processing
                await _action_engine.execute(alert, _broadcast)

    cap.release()
    log.info("Clip processed: %d frames analyzed, %d alerts created", frame_idx, len(alerts_created))

    return {
        "status": "processed",
        "frames_analyzed": frame_idx // sample_interval,
        "total_frames": total_frames,
        "alerts_created": len(alerts_created),
        "alerts": alerts_created,
    }
