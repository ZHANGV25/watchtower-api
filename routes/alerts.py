"""Alert query REST endpoints."""
from __future__ import annotations

import os

import boto3
from fastapi import APIRouter, Depends, HTTPException

import db
from middleware import require_auth

_S3_BUCKET = os.getenv("WATCHTOWER_S3_BUCKET", "watchtower-clips-008524")
_s3 = boto3.client("s3")

router = APIRouter(prefix="/api", tags=["alerts"])


def _presign(s3_key: str) -> str:
    if not s3_key or s3_key.startswith("http"):
        return s3_key
    return _s3.generate_presigned_url(
        "get_object", Params={"Bucket": _S3_BUCKET, "Key": s3_key}, ExpiresIn=3600,
    )


@router.get("/cameras/{camera_id}/alerts")
async def list_alerts(
    camera_id: str,
    limit: int = 50,
    offset: int = 0,
    user: dict = Depends(require_auth),
):
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    raw_alerts = await db.list_alerts(camera_id, limit=limit + 50, offset=offset)
    # Filter out Scene Memory entries (activity log, not real alerts)
    alerts = [a for a in raw_alerts if a.get("rule_id") != "__memory__" and a.get("rule_name") != "Scene Memory"][:limit]
    # Presign S3 keys for frame_path
    for a in alerts:
        if a.get("frame_path"):
            a["frame_path"] = _presign(a["frame_path"])
    total = len([a for a in raw_alerts if a.get("rule_id") != "__memory__" and a.get("rule_name") != "Scene Memory"])
    return {"alerts": alerts, "total": total, "limit": limit, "offset": offset}


@router.get("/alerts/{alert_id}")
async def get_alert(alert_id: str, user: dict = Depends(require_auth)):
    alert = await db.get_alert(alert_id)
    if not alert:
        raise HTTPException(404, "Alert not found")
    return alert


@router.get("/alerts/{alert_id}/clip")
async def get_alert_clip(alert_id: str, user: dict = Depends(require_auth)):
    """Get a presigned URL for the alert's clip video with seek offset."""
    alert = await db.get_alert(alert_id)
    if not alert:
        raise HTTPException(404, "Alert not found")
    clip_key = alert.get("clip_s3_key", "") if isinstance(alert, dict) else getattr(alert, "clip_s3_key", "")
    if not clip_key:
        raise HTTPException(404, "No clip associated with this alert")
    url = _s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": _S3_BUCKET, "Key": clip_key},
        ExpiresIn=3600,
    )
    # Find other alerts from the same clip to determine clip start time
    alert_ts = alert.get("timestamp", 0) if isinstance(alert, dict) else getattr(alert, "timestamp", 0)
    camera_id = alert.get("camera_id", "") if isinstance(alert, dict) else getattr(alert, "camera_id", "")
    # Estimate: alert happened ~midway through clip, seek to 5s before alert moment
    # Clips are max 60s, alert timestamp is absolute. We return the timestamp
    # so frontend can calculate seek position relative to clip start.
    return {"clip_url": url, "s3_key": clip_key, "alert_timestamp": float(alert_ts)}


@router.delete("/cameras/{camera_id}/alerts", status_code=204)
async def clear_alerts(camera_id: str, user: dict = Depends(require_auth)):
    await db.delete_alerts_for_camera(camera_id)
