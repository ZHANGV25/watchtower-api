"""Zone CRUD REST endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import db
from middleware import require_auth
from models import Zone

router = APIRouter(prefix="/api/cameras/{camera_id}/zones", tags=["zones"])


class ZoneCreate(BaseModel):
    name: str
    x: float
    y: float
    width: float
    height: float
    color: str = "#22d3ee"


class ZoneUpdate(BaseModel):
    name: str | None = None
    x: float | None = None
    y: float | None = None
    width: float | None = None
    height: float | None = None
    color: str | None = None


@router.get("")
async def list_zones(camera_id: str, user: dict = Depends(require_auth)):
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    zones = await db.list_zones(camera_id)
    return {"zones": [z.model_dump() for z in zones]}


@router.post("", status_code=201)
async def create_zone(camera_id: str, body: ZoneCreate, user: dict = Depends(require_auth)):
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    zone = Zone(camera_id=camera_id, **body.model_dump())
    await db.create_zone(zone)
    return zone.model_dump()


@router.put("/{zone_id}")
async def update_zone(camera_id: str, zone_id: str, body: ZoneUpdate, user: dict = Depends(require_auth)):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if updates:
        await db.update_zone(zone_id, **updates)
    return {"id": zone_id, **updates}


@router.delete("/{zone_id}", status_code=204)
async def delete_zone(camera_id: str, zone_id: str, user: dict = Depends(require_auth)):
    deleted = await db.delete_zone(zone_id)
    if not deleted:
        raise HTTPException(404, "Zone not found")
