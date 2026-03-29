"""Activity timeline REST endpoint for elder care dashboard."""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta

import boto3
from fastapi import APIRouter, Depends, HTTPException, Query

import db
from middleware import require_auth

_S3_BUCKET = os.getenv("WATCHTOWER_S3_BUCKET", "watchtower-clips-008524")
_s3 = boto3.client("s3")


def _presign(s3_key: str) -> str:
    """Convert an S3 key to a presigned URL. Pass through if already a URL or empty."""
    if not s3_key or s3_key.startswith("http"):
        return s3_key
    return _s3.generate_presigned_url(
        "get_object", Params={"Bucket": _S3_BUCKET, "Key": s3_key}, ExpiresIn=3600,
    )

router = APIRouter(prefix="/api/cameras/{camera_id}/activity", tags=["activity"])


@router.get("")
async def get_activity_timeline(
    camera_id: str,
    date: str | None = Query(None, description="Date in YYYY-MM-DD format (defaults to today)"),
    tz_offset: int = Query(0, description="Client timezone offset in minutes (e.g., -240 for EDT)"),
    limit: int = Query(200, ge=1, le=1000),
    user: dict = Depends(require_auth),
):
    """Return activity log entries for a specific date.

    Uses memory_entries from the database to build a timeline of
    activity observations for the elder care dashboard.
    """
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    # Apply timezone offset so date boundaries match the client's local time
    tz_delta = timedelta(minutes=-tz_offset)

    # Parse date or default to today in client's timezone
    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD.")
    else:
        target_date = (datetime.utcnow() + tz_delta).replace(hour=0, minute=0, second=0, microsecond=0)

    # Compute start/end timestamps for the day in UTC
    local_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    start_time = (local_start - tz_delta).timestamp()
    end_time = (local_start + timedelta(days=1) - tz_delta).timestamp()

    entries = await db.list_memory_entries(camera_id, start_time=start_time, end_time=end_time, limit=limit)

    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "camera_id": camera_id,
        "entries": [
            {
                "id": e.id,
                "timestamp": e.timestamp,
                "time": (datetime.utcfromtimestamp(e.timestamp) + tz_delta).strftime("%I:%M %p").lstrip("0"),
                "summary": e.summary,
                "detection_count": e.detection_count,
                "frame_url": _presign(getattr(e, "frame_url", "") or ""),
            }
            for e in sorted(entries, key=lambda x: x.timestamp)
        ],
        "total": len(entries),
    }


@router.delete("")
async def clear_activity(camera_id: str, user: dict = Depends(require_auth)):
    """Clear all activity entries for this camera."""
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    # Memory entries are stored as alerts with rule_id='__memory__' in DynamoDB,
    # or in the memory_entries table in SQLite. Clear them.
    try:
        # DynamoDB: delete memory alerts
        all_alerts = await db.list_alerts(camera_id, limit=500)
        for a in all_alerts:
            if a.get("rule_id") == "__memory__":
                try:
                    await db.delete_alert(a["id"])
                except Exception:
                    pass
    except Exception:
        pass
    try:
        # SQLite: clear memory entries table
        if hasattr(db, '_db') and db._db:
            await db._db.execute("DELETE FROM memory_entries WHERE camera_id = ?", (camera_id,))
            await db._db.commit()
    except Exception:
        pass
    return {"status": "cleared"}
