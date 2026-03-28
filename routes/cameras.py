"""Camera CRUD REST endpoints."""
from __future__ import annotations

import logging
import secrets
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import db
from middleware import require_auth
from models import Camera, Condition, Rule

log = logging.getLogger("watchtower.cameras")

# In-memory pairing codes (expire after 10 minutes)
_pairing_codes: dict[str, dict] = {}  # code -> {"camera_id": ..., "expires": ...}

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


class CameraCreate(BaseModel):
    name: str = "Unnamed Camera"
    description: str = ""
    location: str = ""


class CameraUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    location: str | None = None


class CameraConnect(BaseModel):
    webrtc_url: str


# ---------------------------------------------------------------------------
# Elder care preset rules — auto-created for every new camera
# ---------------------------------------------------------------------------

_ELDER_CARE_RULES = [
    {
        "name": "Fall Detection",
        "natural_language": "Alert when a person appears to have fallen (lying on the floor)",
        "conditions": [
            {"type": "person_pose", "params": {"pose": "lying"}},
            {"type": "duration", "params": {"seconds": 10}},
        ],
        "severity": "critical",
    },
    {
        "name": "Inactivity Alert",
        "natural_language": "Alert when no person is detected for more than 3 hours during daytime",
        "conditions": [
            {"type": "object_absent", "params": {"class": "person"}},
            {"type": "duration", "params": {"seconds": 10800}},
            {"type": "time_window", "params": {"start_hour": 7, "end_hour": 22}},
        ],
        "severity": "high",
    },
    {
        "name": "Night Wandering",
        "natural_language": "Alert when a person is detected moving between 11pm and 5am",
        "conditions": [
            {"type": "object_present", "params": {"class": "person"}},
            {"type": "time_window", "params": {"start_hour": 23, "end_hour": 5}},
        ],
        "severity": "medium",
    },
    {
        "name": "Visitor Detection",
        "natural_language": "Alert when multiple people are detected in the room",
        "conditions": [
            {"type": "count", "params": {"class": "person", "operator": "gte", "value": 2}},
        ],
        "severity": "low",
    },
    {
        "name": "Emergency - Prolonged Immobility",
        "natural_language": "Alert when a person is lying on the floor with no movement for an extended period",
        "conditions": [
            {"type": "person_pose", "params": {"pose": "lying"}},
            {"type": "stillness", "params": {"seconds": 60}},
            {"type": "duration", "params": {"seconds": 120}},
        ],
        "severity": "critical",
    },
]


async def _create_elder_care_rules(camera_id: str) -> None:
    """Create preset elder care monitoring rules for a new camera."""
    for rule_def in _ELDER_CARE_RULES:
        rule = Rule(
            camera_id=camera_id,
            name=rule_def["name"],
            natural_language=rule_def["natural_language"],
            conditions=[Condition(**c) for c in rule_def["conditions"]],
            severity=rule_def["severity"],
            enabled=True,
        )
        await db.create_rule(rule)
    log.info("Created %d elder care rules for camera %s", len(_ELDER_CARE_RULES), camera_id)


@router.get("")
async def list_cameras(user: dict = Depends(require_auth)):
    cameras = await db.list_cameras()
    return {"cameras": [c.model_dump() for c in cameras]}


@router.post("", status_code=201)
async def create_camera(body: CameraCreate, user: dict = Depends(require_auth)):
    cam = Camera(name=body.name, description=body.description, location=body.location)
    await db.create_camera(cam)
    # Auto-create elder care monitoring rules
    await _create_elder_care_rules(cam.id)
    return cam.model_dump()


@router.get("/{camera_id}")
async def get_camera(camera_id: str, user: dict = Depends(require_auth)):
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    alert_count = await db.count_alerts(camera_id)
    data = cam.model_dump()
    data["alert_count"] = alert_count
    return data


@router.put("/{camera_id}")
async def update_camera(camera_id: str, body: CameraUpdate, user: dict = Depends(require_auth)):
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if updates:
        cam = await db.update_camera(camera_id, **updates)
    return cam.model_dump() if cam else {}


@router.delete("/{camera_id}", status_code=204)
async def delete_camera(camera_id: str, user: dict = Depends(require_auth)):
    deleted = await db.delete_camera(camera_id)
    if not deleted:
        raise HTTPException(404, "Camera not found")


@router.post("/{camera_id}/connect")
async def register_camera_connection(camera_id: str, body: CameraConnect):
    """Camera device calls this on startup to register its WebRTC signaling URL."""
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    await db.update_camera(camera_id, webrtc_url=body.webrtc_url)
    return {"status": "connected", "camera_id": camera_id}


@router.get("/{camera_id}/health")
async def camera_health(camera_id: str, user: dict = Depends(require_auth)):
    """Get camera health status including connection info."""
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    now = time.time()
    last_seen = cam.last_seen or 0
    seconds_ago = now - last_seen if last_seen > 0 else None

    # Determine health
    if seconds_ago is None:
        health = "never_connected"
    elif seconds_ago < 120:  # 2 minutes
        health = "healthy"
    elif seconds_ago < 600:  # 10 minutes
        health = "stale"
    else:
        health = "offline"

    return {
        "camera_id": camera_id,
        "health": health,
        "last_seen": last_seen,
        "seconds_ago": round(seconds_ago) if seconds_ago else None,
        "webrtc_url": cam.webrtc_url,
        "status": cam.status,
    }


@router.post("/{camera_id}/pair")
async def generate_pairing_code(camera_id: str, user: dict = Depends(require_auth)):
    """Generate a 6-digit pairing code for a camera."""
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    code = f"{secrets.randbelow(1000000):06d}"
    _pairing_codes[code] = {
        "camera_id": camera_id,
        "expires": time.time() + 600,  # 10 minutes
    }
    return {"code": code, "expires_in": 600}


@router.post("/pair/{code}")
async def claim_pairing_code(code: str):
    """Camera device claims a pairing code to register itself. No auth required."""
    # Clean expired codes
    now = time.time()
    expired = [k for k, v in _pairing_codes.items() if v["expires"] < now]
    for k in expired:
        del _pairing_codes[k]

    if code not in _pairing_codes:
        raise HTTPException(404, "Invalid or expired pairing code")

    info = _pairing_codes.pop(code)
    camera_id = info["camera_id"]
    cam = await db.get_camera(camera_id)

    return {
        "camera_id": camera_id,
        "camera_name": cam.name if cam else "",
        "status": "paired",
    }
