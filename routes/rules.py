"""Rule CRUD REST endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import db
from middleware import require_auth

router = APIRouter(prefix="/api/cameras/{camera_id}/rules", tags=["rules"])


@router.get("")
async def list_rules(camera_id: str, user: dict = Depends(require_auth)):
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")
    rules = await db.list_rules(camera_id)
    return {"rules": [r.model_dump() for r in rules]}


@router.delete("/{rule_id}", status_code=204)
async def delete_rule(camera_id: str, rule_id: str, user: dict = Depends(require_auth)):
    deleted = await db.delete_rule(rule_id)
    if not deleted:
        raise HTTPException(404, "Rule not found")


@router.patch("/{rule_id}/toggle")
async def toggle_rule(camera_id: str, rule_id: str, user: dict = Depends(require_auth)):
    rules = await db.list_rules(camera_id)
    rule = next((r for r in rules if r.id == rule_id), None)
    if not rule:
        raise HTTPException(404, "Rule not found")
    await db.update_rule(rule_id, enabled=not rule.enabled)
    return {"id": rule_id, "enabled": not rule.enabled}
