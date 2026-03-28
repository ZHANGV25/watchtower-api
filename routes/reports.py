"""Daily and weekly report REST endpoints for elder care dashboard."""
from __future__ import annotations

import time
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse

import db
from middleware import require_auth
from report_generator import ReportGenerator

router = APIRouter(prefix="/api/cameras/{camera_id}/reports", tags=["reports"])

# Shared report generator instance
_report_generator = ReportGenerator()


@router.get("/daily")
async def get_daily_report(
    camera_id: str,
    date: str | None = Query(None, description="Date in YYYY-MM-DD format (defaults to today)"),
    user: dict = Depends(require_auth),
):
    """Generate a structured ADL (Activities of Daily Living) daily report.

    Fetches all memory entries and alerts for the given date and uses
    an LLM to synthesize them into a structured report covering sleep,
    meals, mobility, hydration, visitors, medication, and concerns.
    """
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    # Parse date
    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD.")
    else:
        target_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    date_str = target_date.strftime("%Y-%m-%d")
    start_time = target_date.timestamp()
    end_time = (target_date + timedelta(days=1)).timestamp()

    # Fetch data
    memory_entries = await db.list_memory_entries(camera_id, start_time=start_time, end_time=end_time, limit=500)
    all_alerts = await db.list_alerts(camera_id, limit=500)
    day_alerts = [
        a for a in all_alerts
        if start_time <= a.get("timestamp", 0) < end_time
        and a.get("rule_id") != "__memory__"
    ]

    report = await _report_generator.generate_daily_report(
        activity_entries=memory_entries,
        alerts=day_alerts,
        camera_name=cam.name,
        date=date_str,
    )

    report["camera_id"] = camera_id
    report["room_name"] = cam.name
    return report


@router.get("/weekly")
async def get_weekly_report(
    camera_id: str,
    start_date: str | None = Query(None, description="Start date in YYYY-MM-DD format (defaults to 7 days ago)"),
    user: dict = Depends(require_auth),
):
    """Generate a weekly trends report aggregating 7 days of daily data.

    Generates daily reports for each day in the range, then synthesizes
    them into a weekly trends report covering sleep, meals, mobility,
    visitor frequency, and concerns.
    """
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    # Parse start date
    if start_date:
        try:
            week_start = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD.")
    else:
        week_start = (datetime.now() - timedelta(days=7)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    week_end = week_start + timedelta(days=7)

    # Generate daily reports for each day
    daily_reports: list[dict] = []
    for day_offset in range(7):
        day = week_start + timedelta(days=day_offset)
        day_str = day.strftime("%Y-%m-%d")
        day_start = day.timestamp()
        day_end = (day + timedelta(days=1)).timestamp()

        entries = await db.list_memory_entries(camera_id, start_time=day_start, end_time=day_end, limit=500)
        all_alerts = await db.list_alerts(camera_id, limit=500)
        day_alerts = [
            a for a in all_alerts
            if day_start <= a.get("timestamp", 0) < day_end
            and a.get("rule_id") != "__memory__"
        ]

        if entries or day_alerts:
            report = await _report_generator.generate_daily_report(
                activity_entries=entries,
                alerts=day_alerts,
                camera_name=cam.name,
                date=day_str,
            )
            daily_reports.append(report)
        else:
            daily_reports.append({
                "date": day_str,
                "camera_name": cam.name,
                "activity_count": 0,
                "alert_count": 0,
                "summary": "No activity data available for this day.",
                "concerns": [],
            })

    # Generate weekly summary
    weekly = await _report_generator.generate_weekly_report(
        daily_reports=daily_reports,
        camera_name=cam.name,
        start_date=week_start.strftime("%Y-%m-%d"),
        end_date=(week_end - timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    weekly["camera_id"] = camera_id
    return weekly


@router.get("/export")
async def export_report(
    camera_id: str,
    date: str | None = Query(None, description="Date in YYYY-MM-DD format (defaults to today)"),
    format: str = Query("json", description="Export format: json or text"),
    user: dict = Depends(require_auth),
):
    """Export a daily report in a clean format suitable for clinical review.

    Supports JSON and plain text formats.
    """
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    if format not in ("json", "text"):
        raise HTTPException(400, "Invalid format. Use 'json' or 'text'.")

    # Parse date
    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD.")
    else:
        target_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    date_str = target_date.strftime("%Y-%m-%d")
    start_time = target_date.timestamp()
    end_time = (target_date + timedelta(days=1)).timestamp()

    # Fetch data
    memory_entries = await db.list_memory_entries(camera_id, start_time=start_time, end_time=end_time, limit=500)
    all_alerts = await db.list_alerts(camera_id, limit=500)
    day_alerts = [
        a for a in all_alerts
        if start_time <= a.get("timestamp", 0) < end_time
        and a.get("rule_id") != "__memory__"
    ]

    report = await _report_generator.generate_daily_report(
        activity_entries=memory_entries,
        alerts=day_alerts,
        camera_name=cam.name,
        date=date_str,
    )

    report["camera_id"] = camera_id
    report["room_name"] = cam.name

    if format == "json":
        return report

    # Format as plain text for clinical review
    text = _format_report_text(report)
    return PlainTextResponse(content=text, media_type="text/plain")


def _format_report_text(report: dict) -> str:
    """Format a daily report as readable plain text."""
    lines: list[str] = []
    lines.append(f"DAILY ACTIVITY REPORT")
    lines.append(f"{'=' * 50}")
    lines.append(f"Date: {report.get('date', 'N/A')}")
    lines.append(f"Room: {report.get('room_name', report.get('camera_name', 'N/A'))}")
    lines.append(f"Activity Entries: {report.get('activity_count', 0)}")
    lines.append(f"Alerts: {report.get('alert_count', 0)}")
    lines.append("")

    # Sleep
    sleep = report.get("sleep")
    lines.append("SLEEP")
    lines.append("-" * 30)
    if sleep:
        lines.append(f"  Bed time: {sleep.get('bed_time', 'N/A')}")
        lines.append(f"  Wake time: {sleep.get('wake_time', 'N/A')}")
        lines.append(f"  Duration: {sleep.get('duration_hours', 'N/A')} hours")
        lines.append(f"  Disruptions: {sleep.get('disruptions', 0)}")
    else:
        lines.append("  No sleep data available")
    lines.append("")

    # Meals
    meals = report.get("meals", [])
    lines.append("MEALS")
    lines.append("-" * 30)
    if meals:
        for m in meals:
            lines.append(f"  {m.get('type', 'Meal').capitalize()} at {m.get('time', 'N/A')} ({m.get('duration_minutes', '?')} min)")
    else:
        lines.append("  No meal data available")
    lines.append("")

    # Mobility
    mobility = report.get("mobility")
    lines.append("MOBILITY")
    lines.append("-" * 30)
    if mobility:
        lines.append(f"  Room transitions: {mobility.get('room_transitions', 0)}")
        areas = mobility.get("primary_areas", [])
        lines.append(f"  Primary areas: {', '.join(areas) if areas else 'N/A'}")
    else:
        lines.append("  No mobility data available")
    lines.append("")

    # Hydration
    hydration = report.get("hydration")
    lines.append("HYDRATION")
    lines.append("-" * 30)
    if hydration:
        lines.append(f"  Observations: {hydration.get('observations', 0)}")
        note = hydration.get("note", "")
        if note:
            lines.append(f"  Note: {note}")
    else:
        lines.append("  No hydration data available")
    lines.append("")

    # Visitors
    visitors = report.get("visitors", [])
    lines.append("VISITORS")
    lines.append("-" * 30)
    if visitors:
        for v in visitors:
            lines.append(f"  Visit at {v.get('time', 'N/A')} ({v.get('duration_minutes', '?')} min)")
    else:
        lines.append("  No visitors recorded")
    lines.append("")

    # Medication
    medication = report.get("medication")
    lines.append("MEDICATION")
    lines.append("-" * 30)
    if medication:
        taken = medication.get("taken_on_time")
        if taken is True:
            lines.append("  Taken on time: Yes")
        elif taken is False:
            lines.append("  Taken on time: No")
        else:
            lines.append("  Taken on time: Unknown")
        notes = medication.get("notes", "")
        if notes:
            lines.append(f"  Notes: {notes}")
    else:
        lines.append("  No medication data available")
    lines.append("")

    # Concerns
    concerns = report.get("concerns", [])
    if concerns:
        lines.append("CONCERNS")
        lines.append("-" * 30)
        for c in concerns:
            lines.append(f"  * {c}")
        lines.append("")

    # Summary
    summary = report.get("summary", "")
    if summary:
        lines.append("SUMMARY")
        lines.append("-" * 30)
        lines.append(f"  {summary}")
        lines.append("")

    return "\n".join(lines)
