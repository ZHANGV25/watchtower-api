"""Medication reminder REST endpoints for elder care dashboard.

Medications are stored as rules with a "MED:" prefix in the name,
avoiding the need for a separate DynamoDB table.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import db
from middleware import require_auth
from models import Condition, Rule

router = APIRouter(prefix="/api/cameras/{camera_id}/medications", tags=["medications"])

_MED_PREFIX = "MED: "


class MedicationCreate(BaseModel):
    name: str
    time: str  # HH:MM format (24-hour)
    notes: str = ""


def _is_medication_rule(rule: Rule) -> bool:
    """Check if a rule is a medication reminder."""
    return rule.name.startswith(_MED_PREFIX)


def _rule_to_medication(rule: Rule) -> dict:
    """Convert a medication rule to a medication response dict."""
    # Extract time from time_window condition
    med_time = "00:00"
    for c in rule.conditions:
        if c.type == "time_window":
            hour = c.params.get("start_hour", 0)
            med_time = f"{hour:02d}:00"
            break

    # Extract notes from natural_language
    notes = rule.natural_language

    return {
        "id": rule.id,
        "name": rule.name[len(_MED_PREFIX):],
        "time": med_time,
        "notes": notes,
        "enabled": rule.enabled,
        "created_at": rule.created_at,
    }


@router.get("")
async def list_medications(camera_id: str, user: dict = Depends(require_auth)):
    """List all medication reminders for this camera/room."""
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    rules = await db.list_rules(camera_id)
    medications = [_rule_to_medication(r) for r in rules if _is_medication_rule(r)]

    return {"medications": medications}


@router.post("", status_code=201)
async def create_medication(camera_id: str, body: MedicationCreate, user: dict = Depends(require_auth)):
    """Create a medication reminder.

    Creates a rule with a time_window condition matching the medication
    time. The rule will trigger an alert if no activity is detected
    near the medication area by the scheduled time.
    """
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    if not body.name.strip():
        raise HTTPException(400, "Medication name cannot be empty")

    # Parse the time
    try:
        parts = body.time.split(":")
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError
    except (ValueError, IndexError):
        raise HTTPException(400, "Invalid time format. Use HH:MM (24-hour format).")

    # Create end hour (1 hour window after scheduled time)
    end_hour = (hour + 1) % 24

    # Build the medication rule
    conditions = [
        Condition(type="time_window", params={"start_hour": hour, "end_hour": end_hour}),
        Condition(type="object_absent", params={"class": "person"}),
    ]

    notes_text = f"Take {body.name}"
    if body.notes:
        notes_text += f" - {body.notes}"

    rule = Rule(
        camera_id=camera_id,
        name=f"{_MED_PREFIX}{body.name}",
        natural_language=notes_text,
        conditions=conditions,
        severity="high",
        enabled=True,
    )

    await db.create_rule(rule)

    return _rule_to_medication(rule)


@router.delete("/{rule_id}", status_code=204)
async def delete_medication(camera_id: str, rule_id: str, user: dict = Depends(require_auth)):
    """Delete a medication reminder."""
    # Verify the rule exists and is a medication rule
    rules = await db.list_rules(camera_id)
    rule = next((r for r in rules if r.id == rule_id), None)

    if not rule:
        raise HTTPException(404, "Medication not found")

    if not _is_medication_rule(rule):
        raise HTTPException(400, "The specified rule is not a medication reminder")

    await db.delete_rule(rule_id)
