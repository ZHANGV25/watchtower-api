"""Custom concerns REST endpoint for elder care dashboard.

Allows family members to express monitoring concerns in natural language,
which are converted into structured monitoring rules.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import db
from middleware import require_auth
from rule_parser import RuleParser

router = APIRouter(prefix="/api/cameras/{camera_id}/concerns", tags=["concerns"])

# Shared rule parser instance
_rule_parser = RuleParser()


class ConcernCreate(BaseModel):
    text: str
    severity: str = "medium"


@router.post("")
async def create_concern(camera_id: str, body: ConcernCreate, user: dict = Depends(require_auth)):
    """Convert a natural language concern into a monitoring rule.

    Family members can express concerns like "Mom forgets to drink water"
    and the system will create an appropriate monitoring rule.
    """
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    if not body.text.strip():
        raise HTTPException(400, "Concern text cannot be empty")

    # Get existing zones for context
    zones = await db.list_zones(camera_id)
    zone_names = [z.name for z in zones]

    # Parse the concern into a rule
    result = await _rule_parser.parse(body.text, zone_names, severity=body.severity)
    if result is None:
        raise HTTPException(
            422,
            "Could not convert concern into a monitoring rule. Please try rephrasing.",
        )

    rule, missing_zones = result
    rule.camera_id = camera_id

    # Persist the rule
    await db.create_rule(rule)

    response = rule.model_dump()
    if missing_zones:
        response["_missing_zones"] = missing_zones
        response["_note"] = (
            f"This rule references zones that don't exist yet: {', '.join(missing_zones)}. "
            "The rule will activate once these zones are created."
        )

    return response
