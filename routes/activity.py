"""Activity timeline REST endpoint for elder care dashboard."""
from __future__ import annotations

import time
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query

import db
from middleware import require_auth

router = APIRouter(prefix="/api/cameras/{camera_id}/activity", tags=["activity"])


@router.get("")
async def get_activity_timeline(
    camera_id: str,
    date: str | None = Query(None, description="Date in YYYY-MM-DD format (defaults to today)"),
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

    # Parse date or default to today
    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD.")
    else:
        target_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Compute start/end timestamps for the day
    start_time = target_date.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    end_time = (target_date + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    entries = await db.list_memory_entries(camera_id, start_time=start_time, end_time=end_time, limit=limit)

    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "camera_id": camera_id,
        "entries": [
            {
                "id": e.id,
                "timestamp": e.timestamp,
                "time": datetime.fromtimestamp(e.timestamp).strftime("%I:%M %p").lstrip("0"),
                "summary": e.summary,
                "detection_count": e.detection_count,
                "frame_url": getattr(e, "frame_url", "") or "",
            }
            for e in sorted(entries, key=lambda x: x.timestamp)
        ],
        "total": len(entries),
    }
