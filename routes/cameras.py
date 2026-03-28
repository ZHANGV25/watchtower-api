"""Camera CRUD REST endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import db
from middleware import require_auth
from models import Camera

router = APIRouter(prefix="/api/cameras", tags=["cameras"])


class CameraCreate(BaseModel):
    name: str = "Unnamed Camera"
    description: str = ""
    location: str = ""


class CameraUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    location: str | None = None


@router.get("")
async def list_cameras(user: dict = Depends(require_auth)):
    cameras = await db.list_cameras()
    return {"cameras": [c.model_dump() for c in cameras]}


@router.post("", status_code=201)
async def create_camera(body: CameraCreate, user: dict = Depends(require_auth)):
    cam = Camera(name=body.name, description=body.description, location=body.location)
    await db.create_camera(cam)
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
