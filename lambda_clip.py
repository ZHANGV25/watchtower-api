"""Lambda handler for clip processing.

Triggered by S3 events when a new clip is uploaded. Downloads the clip,
runs YOLO + rules + LLM, creates alerts in DynamoDB, sends notifications.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile

import boto3
import numpy as np

os.environ["WATCHTOWER_DB_BACKEND"] = "dynamodb"
os.environ["WATCHTOWER_NO_CAMERA"] = "1"

log = logging.getLogger("watchtower.lambda_clip")
log.setLevel(logging.INFO)

# Lazy-load heavy modules to improve cold start
_detector = None
_narrator = None
_action_engine = None
_frame_store = None
_face_engine = None


def _init():
    global _detector, _narrator, _action_engine, _frame_store
    if _detector is not None:
        return
    from detector import Detector
    from narrator import Narrator
    from actions import ActionEngine
    from storage import create_frame_store
    _detector = Detector()
    _narrator = Narrator()
    _action_engine = ActionEngine()
    _frame_store = create_frame_store()


def _generate_clip_summary(
    classes: set[str],
    identities: set[str],
    person_count: int,
    alert_count: int,
    clip_duration_s: float,
) -> str:
    """Build a simple text summary from clip detection data (no LLM)."""
    parts: list[str] = []

    # Who was seen
    if identities:
        parts.append(f"{', '.join(sorted(identities))} seen")
    elif person_count > 0:
        parts.append("Person detected")

    # Non-person objects
    objects = sorted(classes - {"person"})
    if objects:
        parts.append(f"Objects: {', '.join(objects)}")

    # Duration
    if clip_duration_s >= 1:
        parts.append(f"Duration: {int(clip_duration_s)}s")

    # Alerts
    if alert_count > 0:
        parts.append(f"{alert_count} alert{'s' if alert_count != 1 else ''} triggered")

    return ". ".join(parts) + "." if parts else "Activity detected."


def handler(event, context):
    """Lambda handler — processes S3 event or direct invocation."""
    _init()

    # Handle S3 trigger
    if "Records" in event:
        for record in event["Records"]:
            s3_info = record.get("s3", {})
            bucket = s3_info.get("bucket", {}).get("name", "")
            key = s3_info.get("object", {}).get("key", "")
            if key.startswith("clips/"):
                # Extract camera_id from key: clips/{camera_id}/{clip_id}.mp4
                parts = key.split("/")
                camera_id = parts[1] if len(parts) >= 3 else "unknown"
                result = asyncio.get_event_loop().run_until_complete(
                    _process_s3_clip(bucket, key, camera_id)
                )
                return result
        return {"status": "no_clips_found"}

    # Handle direct invocation
    camera_id = event.get("camera_id", "")
    s3_key = event.get("s3_key", "")
    capture_time = event.get("timestamp", 0.0)
    bucket = event.get("bucket", os.getenv("WATCHTOWER_S3_BUCKET", "watchtower-clips-008524"))
    if s3_key and camera_id:
        result = asyncio.get_event_loop().run_until_complete(
            _process_s3_clip(bucket, s3_key, camera_id, capture_time=capture_time)
        )
        return result

    return {"status": "error", "message": "No clip to process"}


async def _process_s3_clip(bucket: str, s3_key: str, camera_id: str, capture_time: float = 0.0) -> dict:
    """Download clip from S3 and process it."""
    import cv2
    import database_dynamo as db
    from models import Alert
    from rule_engine import RuleEngine

    s3 = boto3.client("s3")

    # Download to temp file
    local_path = os.path.join(tempfile.gettempdir(), f"clip_{os.path.basename(s3_key)}")
    try:
        s3.download_file(bucket, s3_key, local_path)
    except Exception as e:
        log.error("Failed to download %s/%s: %s", bucket, s3_key, e)
        return {"status": "error", "message": str(e)}

    cap = cv2.VideoCapture(local_path)
    if not cap.isOpened():
        return {"status": "error", "message": f"Cannot open clip: {local_path}"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_fps = cap.get(cv2.CAP_PROP_FPS) or 15

    log.info("Processing clip: %s (%d frames at %.0f fps)", s3_key, total_frames, clip_fps)

    # Load rules and zones for this camera
    rules = await db.list_rules(camera_id)
    zones = await db.list_zones(camera_id)

    if not rules:
        cap.release()
        os.remove(local_path)
        return {"status": "no_rules", "camera_id": camera_id}

    rule_engine = RuleEngine()
    alerts_created = []
    frame_idx = 0
    sample_interval = max(1, int(clip_fps / 5))  # ~5 fps analysis
    import time
    # Use the camera's capture timestamp if provided, otherwise fall back to now
    clip_duration = total_frames / clip_fps if clip_fps > 0 else 0
    base_time = (capture_time - clip_duration) if capture_time > 0 else time.time()

    # Collect detections and sampled frames for LLM analysis
    clip_detections: list[tuple[float, list]] = []
    clip_frames_sampled: list[tuple[np.ndarray, float]] = []  # (frame, timestamp) for LLM fall check
    best_frame = None
    best_frame_det_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % sample_interval != 0:
            continue

        frame_time = base_time + (frame_idx / clip_fps)
        detections = _detector.detect(frame, False)

        # Identify people using face recognition against uploaded reference photo.
        # Runs on every frame with people — clip processor is already at ~5fps.
        person_count = sum(1 for d in detections if d.class_name == "person")
        if person_count >= 1:
            try:
                from face_recognition_engine import FaceRecognitionEngine
                if _face_engine is None:
                    _face_engine_local = FaceRecognitionEngine(data_dir="/tmp/face_data")
                else:
                    _face_engine_local = _face_engine
                if _face_engine_local.has_reference(camera_id):
                    identifications = _face_engine_local.identify_people(camera_id, frame)
                    for det in detections:
                        if det.class_name == "person":
                            det_center_y = det.bbox.y + det.bbox.height / 2
                            for ident in identifications:
                                face_center_y = (ident["location"][0] + ident["location"][2]) / 2
                                if abs(det_center_y - face_center_y) < det.bbox.height:
                                    det.identity = ident["label"]
                                    det.identity_confidence = ident["confidence"]
                                    break
            except (ImportError, OSError):
                pass  # face_recognition not available or filesystem issue

        # Track detections and frames for activity timeline + fall detection
        clip_detections.append((frame_time, detections))
        clip_frames_sampled.append((frame.copy(), frame_time))
        if len(detections) > best_frame_det_count:
            best_frame_det_count = len(detections)
            best_frame = frame.copy()

        fired = rule_engine.evaluate(rules, zones, detections, frame_time)

        for alert_data in fired:
            alert = Alert(
                camera_id=camera_id,
                rule_id=alert_data.rule_id,
                rule_name=alert_data.rule_name,
                severity=alert_data.severity,
                timestamp=frame_time,
                clip_s3_key=s3_key,
                detections=detections[:5],
            )

            # LLM verification
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

            # Save alert to DB
            await db.create_alert(alert, frame_path=alert.frame_path)
            alerts_created.append({"id": alert.id, "rule_name": alert.rule_name, "severity": alert.severity})

            # Push notification
            if _action_engine:
                async def _noop(et, p): pass
                await _action_engine.execute(alert, _noop)

    cap.release()

    # --- LLM concern evaluator: send multiple frames + ALL concerns to Claude ---
    # Claude is the primary decision-maker. YOLO just confirms people/objects exist.
    has_people = any(d.class_name == "person" for ft, dets in clip_detections for d in dets)
    if has_people and _narrator and clip_frames_sampled and rules:
        try:
            import cv2 as _cv2
            import base64

            # Pick ~5 evenly spaced frames from the clip for temporal context
            n = len(clip_frames_sampled)
            indices = [int(i * (n - 1) / min(4, n - 1)) for i in range(min(5, n))]
            frame_images = []
            for idx in indices:
                frm, ts = clip_frames_sampled[idx]
                ok, buf = _cv2.imencode(".jpg", frm, [_cv2.IMWRITE_JPEG_QUALITY, 50])
                if ok:
                    frame_images.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg",
                                   "data": base64.b64encode(buf.tobytes()).decode("ascii")},
                    })

            # Build concern list from all enabled rules
            concern_list = []
            for r in rules:
                if r.enabled:
                    desc = r.natural_language or r.name
                    concern_list.append({"id": r.id, "name": r.name, "description": desc, "severity": r.severity})

            if frame_images and concern_list:
                concerns_text = "\n".join(
                    f"- [{c['id']}] {c['name']} ({c['severity']}): {c['description']}"
                    for c in concern_list
                )

                # Collect YOLO detection summary
                all_detected = set()
                for ft, dets in clip_detections:
                    for d in dets:
                        all_detected.add(d.class_name)

                llm_response = await _narrator._client.messages.create(
                    model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    max_tokens=500,
                    system=f"""You are the monitoring brain for WatchTower, an elder care camera system.

You are shown {len(frame_images)} frames from a {int(total_frames/clip_fps)}s video clip, in chronological order. YOLO detected: {', '.join(sorted(all_detected))}.

The caregiver has set these concerns to monitor for:
{concerns_text}

Analyze the frame sequence and determine which concerns (if any) are triggered.

Respond with ONLY valid JSON (no markdown):
{{
  "triggered": [
    {{"id": "rule_id_here", "description": "Brief explanation of what you see", "confidence": "high/medium/low"}}
  ]
}}

Return an EMPTY triggered array if nothing concerning is happening.

Guidelines:
- Look at the PROGRESSION across frames, not just individual frames
- A fall = person upright then on the floor. Person on bed/couch = NOT a fall.
- Inactivity = no person visible in ANY frame (the room is empty)
- Night wandering = person moving AND it's nighttime (check if scene looks dark/nighttime)
- Be conservative — only trigger if you genuinely see the concern happening
- "confidence" helps prioritize: high = clearly happening, medium = likely, low = possible""",
                    messages=[{"role": "user", "content": [
                        *frame_images,
                        {"type": "text", "text": "Which concerns, if any, are triggered in this clip?"},
                    ]}],
                )

                raw = llm_response.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()
                result = json.loads(raw)

                for triggered in result.get("triggered", []):
                    rule_id = triggered.get("id", "")
                    description = triggered.get("description", "")
                    confidence = triggered.get("confidence", "medium")

                    # Find the matching rule
                    rule = next((r for r in rules if r.id == rule_id), None)
                    if not rule:
                        # LLM might return "llm_fall_detection" or similar
                        rule_name = triggered.get("name", "Concern Triggered")
                        severity = "medium"
                    else:
                        rule_name = rule.name
                        severity = rule.severity

                    # Skip low confidence unless critical severity
                    if confidence == "low" and severity not in ("critical", "high"):
                        continue

                    # Check cooldown — don't duplicate alerts from rule engine
                    existing = [a for a in alerts_created if a.get("rule_name") == rule_name]
                    if existing:
                        continue

                    concern_alert = Alert(
                        camera_id=camera_id,
                        rule_id=rule_id or "llm_concern",
                        rule_name=rule_name,
                        severity=severity,
                        timestamp=base_time,
                        clip_s3_key=s3_key,
                        narration=description,
                        detections=[d for ft, dets in clip_detections for d in dets if d.class_name == "person"][:3],
                    )
                    if _frame_store and best_frame is not None:
                        frame_bytes = _cv2.imencode(".jpg", best_frame, [_cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                        path = await _frame_store.save_frame(concern_alert.id, frame_bytes)
                        concern_alert.frame_path = path
                    await db.create_alert(concern_alert, frame_path=concern_alert.frame_path)
                    alerts_created.append({"id": concern_alert.id, "rule_name": rule_name, "severity": severity})
                    log.info("LLM concern triggered: %s — %s (confidence: %s)", rule_name, description, confidence)

        except Exception as e:
            log.warning("LLM concern evaluation failed: %s", e)

    # --- Create activity timeline (memory) entry for this clip ---
    try:
        from models import MemoryEntry

        all_classes: set[str] = set()
        all_identities: set[str] = set()
        total_person_detections = 0
        for _ft, dets in clip_detections:
            for det in dets:
                all_classes.add(det.class_name)
                if det.class_name == "person":
                    total_person_detections += 1
                    if det.identity:
                        all_identities.add(det.identity)

        if total_person_detections > 0:
            # Try LLM-based summary first using narrator + the best frame
            summary = ""
            if _narrator and best_frame is not None:
                try:
                    # Gather detections from the best frame for context
                    best_dets = max(clip_detections, key=lambda x: len(x[1]))[1]
                    summary = await _narrator.narrate_scene(best_frame, best_dets)
                except Exception as e:
                    log.warning("LLM narration for memory entry failed, using fallback: %s", e)

            # Fallback: simple text summary from detection data
            if not summary:
                summary = _generate_clip_summary(
                    all_classes, all_identities, total_person_detections, len(alerts_created),
                    total_frames / clip_fps,
                )

            # Save snapshot frame to S3 for visual proof
            frame_url = ""
            if best_frame is not None and _frame_store:
                try:
                    frame_bytes = cv2.imencode(".jpg", best_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                    frame_key = f"timeline_{camera_id}_{int(base_time)}"
                    frame_url = await _frame_store.save_frame(frame_key, frame_bytes)
                except Exception as e:
                    log.warning("Failed to save timeline frame: %s", e)

            entry = MemoryEntry(
                timestamp=base_time,
                summary=summary,
                detection_count=total_person_detections,
                frame_url=frame_url,
            )
            await db.create_memory_entry(camera_id, entry)
            log.info("Created memory entry for clip: %s", entry.summary[:80])
    except Exception as e:
        log.error("Failed to create memory entry: %s", e)

    os.remove(local_path)

    log.info("Processed: %d frames, %d alerts", frame_idx // sample_interval, len(alerts_created))
    return {
        "status": "processed",
        "camera_id": camera_id,
        "frames_analyzed": frame_idx // sample_interval,
        "alerts_created": len(alerts_created),
        "alerts": alerts_created,
    }
