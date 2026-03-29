"""Investigation REST endpoint for elder care dashboard.

Allows family members to ask natural language questions about what
their loved one has been doing, answered using memory entries and alerts.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta

import anthropic
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import db
from middleware import require_auth

log = logging.getLogger("watchtower.investigate")

router = APIRouter(prefix="/api/cameras/{camera_id}/investigate", tags=["investigate"])

_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"

_INVESTIGATE_PROMPT = """\
You are WatchTower's elder care investigation assistant. You have access to a scene memory log — timestamped summaries of what a camera has observed throughout the day in an elderly person's room.

Given the memory log and the user's question, provide a clear, concise, warm answer. Reference specific times when relevant. If you can't answer from the available memory, say so honestly.

## Scene Memory Log
{memory_log}

## Recent Alerts
{alert_log}

Answer the user's question based on the above context. Reference specific times using **HH:MM** format (bold). Keep your answer to 2-4 sentences. Be warm and reassuring when things are normal. Be factual and calm when reporting concerns."""


class InvestigateRequest(BaseModel):
    question: str
    time_range_minutes: int = 120  # Default: last 2 hours
    tz_offset: int = 0  # Client timezone offset in minutes


@router.post("")
async def investigate(camera_id: str, body: InvestigateRequest, user: dict = Depends(require_auth)):
    """Answer a natural language question about what happened in the room.

    Uses memory entries (activity log) and alerts from the database to
    provide context for the LLM to answer questions like "What did Mom
    do today?" or "When did she last eat?"
    """
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    if not body.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    now = time.time()
    start_time = now - (body.time_range_minutes * 60)

    # Fetch memory entries for the time range
    entries = await db.list_memory_entries(
        camera_id, start_time=start_time, end_time=now, limit=200
    )

    # Format memory log (convert to client timezone)
    tz_delta = timedelta(minutes=-body.tz_offset)
    if entries:
        memory_lines = []
        for e in sorted(entries, key=lambda x: x.timestamp):
            t = (datetime.utcfromtimestamp(e.timestamp) + tz_delta).strftime("%I:%M %p").lstrip("0")
            memory_lines.append(f"[{t}] {e.summary}")
        memory_log = "\n".join(memory_lines)
    else:
        memory_log = "(No activity recorded in this time range)"

    # Fetch recent alerts
    all_alerts = await db.list_alerts(camera_id, limit=50)
    recent_alerts = [
        a for a in all_alerts
        if a.get("timestamp", 0) >= start_time
        and a.get("rule_id") != "__memory__"
    ]

    if recent_alerts:
        alert_lines = []
        for a in sorted(recent_alerts, key=lambda x: x.get("timestamp", 0)):
            t = (datetime.utcfromtimestamp(a.get("timestamp", 0)) + tz_delta).strftime("%I:%M %p").lstrip("0")
            alert_lines.append(
                f"[{t}] {a.get('rule_name', 'Unknown')} ({a.get('severity', 'medium')}): {a.get('narration', '')}"
            )
        alert_log = "\n".join(alert_lines)
    else:
        alert_log = "(No recent alerts)"

    # Build prompt
    prompt = _INVESTIGATE_PROMPT.replace("{memory_log}", memory_log).replace("{alert_log}", alert_log)

    try:
        client = anthropic.AsyncAnthropicBedrock(
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
        )
        response = await client.messages.create(
            model=_BEDROCK_MODEL,
            max_tokens=400,
            system=prompt,
            messages=[{"role": "user", "content": body.question}],
        )
        answer = response.content[0].text.strip()
    except Exception as e:
        log.error("Investigation failed: %s", e)
        answer = "Sorry, I couldn't process that question right now. Please try again."

    return {
        "answer": answer,
        "question": body.question,
        "relevant_frames": [],  # No frames available in REST mode (camera not connected)
        "memory_entries_used": len(entries),
        "alerts_used": len(recent_alerts),
    }
