"""Room status summary REST endpoint for elder care dashboard."""
from __future__ import annotations

import time
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException

import db
from middleware import require_auth

router = APIRouter(prefix="/api/cameras/{camera_id}/status", tags=["status"])


@router.get("")
async def get_status(camera_id: str, user: dict = Depends(require_auth)):
    """Return current status of the monitored room.

    Checks recent alerts and memory entries to determine overall status:
    - critical: Critical alert in last hour
    - alert: High-severity alert in last 2 hours
    - warning: No memory entries in last 3 hours
    - good: Normal activity detected
    """
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    now = time.time()
    one_hour_ago = now - 3600
    two_hours_ago = now - 7200
    three_hours_ago = now - 10800
    twenty_four_hours_ago = now - 86400

    # Get recent alerts (last 24h)
    all_alerts = await db.list_alerts(camera_id, limit=200)
    recent_alerts = [
        a for a in all_alerts
        if a.get("timestamp", 0) >= twenty_four_hours_ago
        and a.get("rule_id") != "__memory__"
    ]

    # Count alerts today (midnight to now)
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    alerts_today = [a for a in recent_alerts if a.get("timestamp", 0) >= today_start]

    # Get recent memory entries (last 3 hours)
    memory_entries = await db.list_memory_entries(camera_id, start_time=three_hours_ago, end_time=now, limit=10)

    # Find last activity (most recent memory entry)
    all_memory = await db.list_memory_entries(camera_id, start_time=twenty_four_hours_ago, end_time=now, limit=1)
    last_activity = all_memory[0].timestamp if all_memory else None

    # Cross-room spatial awareness: check other cameras for recent activity
    all_cameras = await db.list_cameras()
    other_activity = None
    for other_cam in all_cameras:
        if other_cam.id == camera_id:
            continue
        other_entries = await db.list_memory_entries(
            other_cam.id, start_time=three_hours_ago, end_time=now, limit=10
        )
        if other_entries:
            latest = max(other_entries, key=lambda e: e.timestamp)
            if other_activity is None or latest.timestamp > other_activity["timestamp"]:
                other_activity = {"room": other_cam.name, "timestamp": latest.timestamp}

    # Find last alert
    last_alert = None
    if recent_alerts:
        last_alert = {
            "id": recent_alerts[0].get("id"),
            "rule_name": recent_alerts[0].get("rule_name"),
            "severity": recent_alerts[0].get("severity"),
            "timestamp": recent_alerts[0].get("timestamp"),
            "narration": recent_alerts[0].get("narration", ""),
        }

    # Determine status level and text
    status_level = "good"
    status_text = _generate_good_status(cam.name, last_activity, now)

    # Check for critical alerts in last hour
    critical_recent = [
        a for a in recent_alerts
        if a.get("severity") == "critical" and a.get("timestamp", 0) >= one_hour_ago
    ]
    if critical_recent:
        status_level = "critical"
        alert = critical_recent[0]
        mins_ago = int((now - alert.get("timestamp", now)) / 60)
        status_text = f"Alert: {alert.get('rule_name', 'Critical event')} detected {mins_ago} minute{'s' if mins_ago != 1 else ''} ago"
    else:
        # Check for high-severity alerts in last 2 hours
        high_recent = [
            a for a in recent_alerts
            if a.get("severity") == "high" and a.get("timestamp", 0) >= two_hours_ago
        ]
        if high_recent:
            status_level = "alert"
            alert = high_recent[0]
            mins_ago = int((now - alert.get("timestamp", now)) / 60)
            status_text = f"Alert: {alert.get('rule_name', 'Event')} detected {mins_ago} minute{'s' if mins_ago != 1 else ''} ago"
        elif not memory_entries:
            if other_activity:
                # Person is active in another room — not concerning
                status_level = "good"
                mins_ago = int((now - other_activity["timestamp"]) / 60)
                if mins_ago < 5:
                    status_text = f"No activity here. Currently in {other_activity['room']}."
                else:
                    status_text = f"No recent activity. Last seen in {other_activity['room']} {mins_ago} min ago."
            else:
                # No activity across ANY room — genuinely concerning
                status_level = "warning"
                if last_activity:
                    hours_ago = (now - last_activity) / 3600
                    if hours_ago < 1:
                        mins = int((now - last_activity) / 60)
                        status_text = f"No activity detected for {mins} minutes"
                    else:
                        status_text = f"No activity detected for {int(hours_ago)} hour{'s' if int(hours_ago) != 1 else ''}"
                else:
                    status_text = "No recent activity detected"

    return {
        "camera_id": camera_id,
        "camera_name": cam.name,
        "status_level": status_level,
        "status_text": status_text,
        "last_activity": last_activity,
        "last_alert": last_alert,
        "alert_count_today": len(alerts_today),
    }


def _generate_good_status(camera_name: str, last_activity: float | None, now: float) -> str:
    """Generate a friendly status text for normal conditions."""
    if not last_activity:
        return "Monitoring active, no activity recorded yet"

    mins_ago = int((now - last_activity) / 60)
    if mins_ago < 5:
        return "Activity detected just now. Everything looks normal."
    elif mins_ago < 30:
        return f"Activity detected {mins_ago} minutes ago. Everything looks normal."
    elif mins_ago < 60:
        return "Activity detected in the last hour. Everything looks fine."
    else:
        hours = int(mins_ago / 60)
        return f"Last activity was {hours} hour{'s' if hours != 1 else ''} ago."
