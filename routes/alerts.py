"""Alert query REST endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

import db
from middleware import require_auth

router = APIRouter(prefix="/api", tags=["alerts"])


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
    alerts = await db.list_alerts(camera_id, limit=limit, offset=offset)
    total = await db.count_alerts(camera_id)
    return {"alerts": alerts, "total": total, "limit": limit, "offset": offset}


@router.get("/alerts/{alert_id}")
async def get_alert(alert_id: str, user: dict = Depends(require_auth)):
    alert = await db.get_alert(alert_id)
    if not alert:
        raise HTTPException(404, "Alert not found")
    return alert


@router.delete("/cameras/{camera_id}/alerts", status_code=204)
async def clear_alerts(camera_id: str, user: dict = Depends(require_auth)):
    await db.delete_alerts_for_camera(camera_id)
