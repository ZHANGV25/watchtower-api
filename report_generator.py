"""ADL (Activities of Daily Living) report generator for elder care.

Uses Claude Sonnet via Bedrock to synthesize activity logs and alerts
into structured daily and weekly reports.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta

import anthropic

from models import Alert, MemoryEntry

log = logging.getLogger("watchtower.report_generator")

_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"

_DAILY_REPORT_PROMPT = """\
You are an elder care activity analyzer for WatchTower. Given a day's worth of scene memory entries \
and alerts from a single room camera, produce a structured Activities of Daily Living (ADL) report.

## Instructions
- Analyze the timestamps and summaries to infer daily patterns
- If data is sparse, make reasonable inferences but note uncertainty
- All times should be in 12-hour format (e.g., "7:15am")
- Be warm and family-friendly in tone (this report is for family members)
- If a category has no data, use null or reasonable defaults
- The "concerns" array should list anything noteworthy for family members

## Output format
Return ONLY valid JSON (no markdown, no explanation):
{
  "sleep": {"bed_time": "10:45pm", "wake_time": "6:30am", "duration_hours": 7.75, "disruptions": 0},
  "meals": [{"time": "7:15am", "duration_minutes": 25, "type": "breakfast"}],
  "mobility": {"room_transitions": 0, "primary_areas": ["kitchen", "living room"]},
  "hydration": {"observations": 0, "note": "No hydration data available"},
  "visitors": [{"time": "3:00pm", "duration_minutes": 45}],
  "medication": {"taken_on_time": null, "notes": "No medication data available"},
  "concerns": [],
  "summary": "A one-paragraph warm summary of the day for family members."
}"""

_WEEKLY_REPORT_PROMPT = """\
You are an elder care trend analyzer for WatchTower. Given a week's worth of daily activity summaries, \
produce a weekly trends report highlighting patterns, improvements, and concerns.

## Instructions
- Look for trends across the week (improving, declining, stable)
- Be specific about numbers and patterns
- Flag anything that warrants attention from family or caregivers
- Provide actionable recommendations
- Be warm and constructive in tone

## Output format
Return ONLY valid JSON (no markdown, no explanation):
{
  "trends": {
    "sleep_avg_hours": 7.2,
    "sleep_trend": "stable",
    "meal_consistency": "2-3 meals/day",
    "mobility_trend": "stable",
    "visitor_frequency": "2 visits/week",
    "concerns": []
  },
  "daily_summaries": ["Monday: ...", "Tuesday: ..."],
  "recommendation": "A brief recommendation paragraph for family members."
}"""


class ReportGenerator:
    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropicBedrock(
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
        )

    async def generate_daily_report(
        self,
        activity_entries: list[MemoryEntry],
        alerts: list[dict],
        camera_name: str,
        date: str,
    ) -> dict:
        """Generate a structured ADL daily report from activity entries and alerts.

        Args:
            activity_entries: Memory entries for the day
            alerts: Alert dicts for the day
            camera_name: Name of the camera/room
            date: Date string (YYYY-MM-DD)

        Returns:
            Structured daily report dict
        """
        # Format memory entries
        memory_lines: list[str] = []
        for e in sorted(activity_entries, key=lambda x: x.timestamp):
            t = datetime.fromtimestamp(e.timestamp).strftime("%I:%M%p").lstrip("0").lower()
            memory_lines.append(f"[{t}] {e.summary} (detections: {e.detection_count})")

        # Format alerts
        alert_lines: list[str] = []
        for a in sorted(alerts, key=lambda x: x.get("timestamp", 0)):
            t = datetime.fromtimestamp(a.get("timestamp", 0)).strftime("%I:%M%p").lstrip("0").lower()
            alert_lines.append(f"[{t}] {a.get('rule_name', 'Unknown')} ({a.get('severity', 'medium')}): {a.get('narration', '')}")

        memory_log = "\n".join(memory_lines) if memory_lines else "(No activity recorded)"
        alert_log = "\n".join(alert_lines) if alert_lines else "(No alerts)"

        user_content = (
            f"Room: {camera_name}\n"
            f"Date: {date}\n\n"
            f"## Activity Log ({len(activity_entries)} entries)\n{memory_log}\n\n"
            f"## Alerts ({len(alerts)} alerts)\n{alert_log}"
        )

        try:
            response = await self._client.messages.create(
                model=_BEDROCK_MODEL,
                max_tokens=2048,
                system=_DAILY_REPORT_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )

            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1])

            report = json.loads(raw)

            # Wrap with metadata
            return {
                "date": date,
                "camera_name": camera_name,
                "activity_count": len(activity_entries),
                "alert_count": len(alerts),
                **report,
            }

        except Exception as e:
            log.error("Daily report generation failed: %s", e)
            return {
                "date": date,
                "camera_name": camera_name,
                "activity_count": len(activity_entries),
                "alert_count": len(alerts),
                "sleep": None,
                "meals": [],
                "mobility": None,
                "hydration": None,
                "visitors": [],
                "medication": None,
                "concerns": ["Report generation failed — insufficient data or service error"],
                "summary": "Unable to generate report. Please try again later.",
            }

    async def generate_weekly_report(
        self,
        daily_reports: list[dict],
        camera_name: str,
        start_date: str,
        end_date: str,
    ) -> dict:
        """Generate a weekly trends report from daily reports.

        Args:
            daily_reports: List of daily report dicts
            camera_name: Name of the camera/room
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Structured weekly report dict
        """
        # Build summaries of each daily report for the LLM
        daily_summaries: list[str] = []
        for report in daily_reports:
            date = report.get("date", "unknown")
            summary = report.get("summary", "No data")
            concerns = report.get("concerns", [])
            concern_text = f" Concerns: {', '.join(concerns)}" if concerns else ""
            daily_summaries.append(f"{date}: {summary}{concern_text}")

        summaries_text = "\n".join(daily_summaries) if daily_summaries else "(No daily reports available)"

        user_content = (
            f"Room: {camera_name}\n"
            f"Week: {start_date} to {end_date}\n\n"
            f"## Daily Summaries\n{summaries_text}"
        )

        try:
            response = await self._client.messages.create(
                model=_BEDROCK_MODEL,
                max_tokens=2048,
                system=_WEEKLY_REPORT_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )

            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1])

            report = json.loads(raw)

            return {
                "start_date": start_date,
                "end_date": end_date,
                "camera_name": camera_name,
                **report,
            }

        except Exception as e:
            log.error("Weekly report generation failed: %s", e)
            return {
                "start_date": start_date,
                "end_date": end_date,
                "camera_name": camera_name,
                "trends": {
                    "sleep_avg_hours": None,
                    "sleep_trend": "unknown",
                    "meal_consistency": "unknown",
                    "mobility_trend": "unknown",
                    "visitor_frequency": "unknown",
                    "concerns": ["Report generation failed"],
                },
                "daily_summaries": [],
                "recommendation": "Unable to generate weekly report. Please try again later.",
            }
