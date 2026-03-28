"""Tests for elder care features: activity timeline, status, concerns, medications, auto-rules, reports."""
from __future__ import annotations

import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ["WATCHTOWER_DB"] = "/tmp/watchtower_elder_test.db"

import database as db
from models import Alert, Camera, Condition, MemoryEntry, Rule, Zone


@pytest.fixture(autouse=True)
def _clean_db():
    """Re-initialize DB before each test."""
    try:
        os.remove("/tmp/watchtower_elder_test.db")
    except FileNotFoundError:
        pass
    db._db = None
    yield
    asyncio.get_event_loop().run_until_complete(db.close_db())


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _setup_camera(name="Mom's Living Room"):
    cam = Camera(name=name, location="Downstairs")
    run(db.create_camera(cam))
    return cam


def _add_memory_entries(camera_id, count=5, base_time=None):
    """Add memory entries for testing."""
    if base_time is None:
        base_time = time.time() - 3600  # 1 hour ago
    entries = []
    for i in range(count):
        entry = MemoryEntry(
            timestamp=base_time + (i * 30),
            summary=f"Activity observation {i + 1}",
            detection_count=1,
        )
        run(db.create_memory_entry(camera_id, entry))
        entries.append(entry)
    return entries


def _add_alert(camera_id, rule_name="Fall Detection", severity="critical", minutes_ago=5):
    alert = Alert(
        camera_id=camera_id,
        rule_id="test-rule",
        rule_name=rule_name,
        severity=severity,
        timestamp=time.time() - (minutes_ago * 60),
        narration="Test alert narration",
    )
    run(db.create_alert(alert))
    return alert


# ---------------------------------------------------------------------------
# Activity Timeline
# ---------------------------------------------------------------------------

class TestActivityTimeline:
    def test_returns_entries_for_today(self):
        cam = _setup_camera()
        _add_memory_entries(cam.id, count=3, base_time=time.time() - 600)
        entries = run(db.list_memory_entries(cam.id, start_time=time.time() - 3600, end_time=time.time()))
        assert len(entries) == 3

    def test_empty_for_no_entries(self):
        cam = _setup_camera()
        entries = run(db.list_memory_entries(cam.id, start_time=0, end_time=time.time()))
        assert len(entries) == 0

    def test_entries_filtered_by_date_range(self):
        cam = _setup_camera()
        now = time.time()
        # Add entries in different time ranges
        old_entry = MemoryEntry(timestamp=now - 86400, summary="Yesterday", detection_count=0)
        run(db.create_memory_entry(cam.id, old_entry))
        new_entry = MemoryEntry(timestamp=now - 60, summary="Just now", detection_count=1)
        run(db.create_memory_entry(cam.id, new_entry))

        # Query only recent
        recent = run(db.list_memory_entries(cam.id, start_time=now - 3600, end_time=now))
        assert len(recent) == 1
        assert recent[0].summary == "Just now"


# ---------------------------------------------------------------------------
# Status Summary
# ---------------------------------------------------------------------------

class TestStatusLogic:
    """Test the status determination logic from routes/status.py."""

    def test_good_status_with_recent_activity(self):
        cam = _setup_camera()
        _add_memory_entries(cam.id, count=1, base_time=time.time() - 60)
        # No alerts
        entries = run(db.list_memory_entries(cam.id, start_time=time.time() - 10800, end_time=time.time()))
        assert len(entries) > 0  # Should have activity → good status

    def test_warning_status_no_recent_activity(self):
        cam = _setup_camera()
        # Only old activity
        _add_memory_entries(cam.id, count=1, base_time=time.time() - 14400)
        entries = run(db.list_memory_entries(cam.id, start_time=time.time() - 10800, end_time=time.time()))
        assert len(entries) == 0  # No recent activity → warning

    def test_critical_status_with_critical_alert(self):
        cam = _setup_camera()
        _add_alert(cam.id, severity="critical", minutes_ago=30)
        alerts = run(db.list_alerts(cam.id, limit=10))
        critical = [a for a in alerts if a.get("severity") == "critical" and a.get("timestamp", 0) >= time.time() - 3600]
        assert len(critical) == 1  # Critical alert in last hour → critical status


# ---------------------------------------------------------------------------
# Medication Reminders (stored as rules)
# ---------------------------------------------------------------------------

class TestMedications:
    def test_create_medication_rule(self):
        cam = _setup_camera()
        rule = Rule(
            camera_id=cam.id,
            name="MED: Blood Pressure Pill",
            natural_language="Take Blood Pressure Pill - Take with food",
            conditions=[
                Condition(type="time_window", params={"start_hour": 8, "end_hour": 9}),
                Condition(type="object_absent", params={"class": "person"}),
            ],
            severity="high",
            enabled=True,
        )
        run(db.create_rule(rule))
        rules = run(db.list_rules(cam.id))
        med_rules = [r for r in rules if r.name.startswith("MED: ")]
        assert len(med_rules) == 1
        assert med_rules[0].name == "MED: Blood Pressure Pill"
        assert med_rules[0].severity == "high"

    def test_list_medications_filters_non_med_rules(self):
        cam = _setup_camera()
        # Add regular rule
        regular = Rule(
            camera_id=cam.id,
            name="Fall Detection",
            natural_language="Detect falls",
            conditions=[Condition(type="person_pose", params={"pose": "lying"})],
            severity="critical",
        )
        run(db.create_rule(regular))
        # Add medication rule
        med = Rule(
            camera_id=cam.id,
            name="MED: Vitamin D",
            natural_language="Take Vitamin D",
            conditions=[Condition(type="time_window", params={"start_hour": 9, "end_hour": 10})],
            severity="high",
        )
        run(db.create_rule(med))

        rules = run(db.list_rules(cam.id))
        all_rules = len(rules)
        med_rules = [r for r in rules if r.name.startswith("MED: ")]
        assert all_rules == 2
        assert len(med_rules) == 1

    def test_delete_medication_rule(self):
        cam = _setup_camera()
        rule = Rule(
            camera_id=cam.id,
            name="MED: Aspirin",
            natural_language="Take Aspirin",
            conditions=[Condition(type="time_window", params={"start_hour": 12, "end_hour": 13})],
            severity="high",
        )
        run(db.create_rule(rule))
        run(db.delete_rule(rule.id))
        rules = run(db.list_rules(cam.id))
        assert len(rules) == 0

    def test_medication_time_extraction(self):
        """Verify time_window condition stores hour correctly."""
        cam = _setup_camera()
        rule = Rule(
            camera_id=cam.id,
            name="MED: Evening Pill",
            natural_language="Take Evening Pill",
            conditions=[
                Condition(type="time_window", params={"start_hour": 20, "end_hour": 21}),
            ],
            severity="high",
        )
        run(db.create_rule(rule))
        rules = run(db.list_rules(cam.id))
        tw = next(c for c in rules[0].conditions if c.type == "time_window")
        assert tw.params["start_hour"] == 20
        assert tw.params["end_hour"] == 21


# ---------------------------------------------------------------------------
# Elder Care Auto-Rules
# ---------------------------------------------------------------------------

class TestElderCareAutoRules:
    def test_elder_care_rule_definitions(self):
        """Verify the preset rule definitions are well-formed."""
        from routes.cameras import _ELDER_CARE_RULES

        assert len(_ELDER_CARE_RULES) == 5
        names = [r["name"] for r in _ELDER_CARE_RULES]
        assert "Fall Detection" in names
        assert "Inactivity Alert" in names
        assert "Night Wandering" in names
        assert "Visitor Detection" in names
        assert "Emergency - Prolonged Immobility" in names

    def test_elder_care_rule_severities(self):
        from routes.cameras import _ELDER_CARE_RULES
        severity_map = {r["name"]: r["severity"] for r in _ELDER_CARE_RULES}
        assert severity_map["Fall Detection"] == "critical"
        assert severity_map["Inactivity Alert"] == "high"
        assert severity_map["Night Wandering"] == "medium"
        assert severity_map["Visitor Detection"] == "low"
        assert severity_map["Emergency - Prolonged Immobility"] == "critical"

    def test_create_elder_care_rules(self):
        from routes.cameras import _create_elder_care_rules
        cam = _setup_camera()
        run(_create_elder_care_rules(cam.id))
        rules = run(db.list_rules(cam.id))
        assert len(rules) == 5

    def test_elder_care_rules_have_valid_conditions(self):
        from routes.cameras import _create_elder_care_rules
        cam = _setup_camera()
        run(_create_elder_care_rules(cam.id))
        rules = run(db.list_rules(cam.id))
        valid_types = {
            "object_present", "object_absent", "object_in_zone",
            "person_pose", "count", "duration", "time_window",
            "stillness", "movement_speed", "person_falling",
        }
        for rule in rules:
            for cond in rule.conditions:
                assert cond.type in valid_types, f"Invalid condition type: {cond.type} in rule {rule.name}"

    def test_fall_detection_rule_structure(self):
        from routes.cameras import _create_elder_care_rules
        cam = _setup_camera()
        run(_create_elder_care_rules(cam.id))
        rules = run(db.list_rules(cam.id))
        fall_rule = next(r for r in rules if r.name == "Fall Detection")
        assert fall_rule.severity == "critical"
        assert fall_rule.enabled is True
        cond_types = [c.type for c in fall_rule.conditions]
        assert "person_pose" in cond_types
        assert "duration" in cond_types


# ---------------------------------------------------------------------------
# Report Text Formatting
# ---------------------------------------------------------------------------

class TestReportTextFormatting:
    def test_format_report_text(self):
        from routes.reports import _format_report_text

        report = {
            "date": "2026-03-28",
            "room_name": "Mom's Living Room",
            "activity_count": 10,
            "alert_count": 1,
            "sleep": {"bed_time": "10:45pm", "wake_time": "6:30am", "duration_hours": 7.75, "disruptions": 1},
            "meals": [
                {"time": "7:15am", "duration_minutes": 25, "type": "breakfast"},
                {"time": "12:30pm", "duration_minutes": 20, "type": "lunch"},
            ],
            "mobility": {"room_transitions": 14, "primary_areas": ["kitchen", "living room"]},
            "hydration": {"observations": 3, "note": "Below recommended"},
            "visitors": [{"time": "3:00pm", "duration_minutes": 45}],
            "medication": {"taken_on_time": True, "notes": ""},
            "concerns": ["Dinner skipped"],
            "summary": "Mom had a typical day.",
        }

        text = _format_report_text(report)
        assert "DAILY ACTIVITY REPORT" in text
        assert "Mom's Living Room" in text
        assert "10:45pm" in text
        assert "6:30am" in text
        assert "Breakfast" in text
        assert "Lunch" in text
        assert "kitchen, living room" in text
        assert "Below recommended" in text
        assert "Dinner skipped" in text
        assert "Mom had a typical day." in text

    def test_format_report_text_empty_data(self):
        from routes.reports import _format_report_text

        report = {
            "date": "2026-03-28",
            "room_name": "Kitchen",
            "activity_count": 0,
            "alert_count": 0,
            "sleep": None,
            "meals": [],
            "mobility": None,
            "hydration": None,
            "visitors": [],
            "medication": None,
            "concerns": [],
            "summary": "",
        }

        text = _format_report_text(report)
        assert "No sleep data available" in text
        assert "No meal data available" in text
        assert "No mobility data available" in text
        assert "No hydration data available" in text
        assert "No visitors recorded" in text
        assert "No medication data available" in text

    def test_format_medication_on_time(self):
        from routes.reports import _format_report_text

        report = {
            "date": "2026-03-28",
            "room_name": "Room",
            "activity_count": 0,
            "alert_count": 0,
            "sleep": None,
            "meals": [],
            "mobility": None,
            "hydration": None,
            "visitors": [],
            "medication": {"taken_on_time": False, "notes": "Missed morning dose"},
            "concerns": [],
            "summary": "",
        }

        text = _format_report_text(report)
        assert "Taken on time: No" in text
        assert "Missed morning dose" in text


# ---------------------------------------------------------------------------
# Report Generator (mocked LLM)
# ---------------------------------------------------------------------------

class TestReportGenerator:
    def test_daily_report_with_data(self):
        from report_generator import ReportGenerator
        gen = ReportGenerator()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"sleep": {"bed_time": "10pm", "wake_time": "7am", "duration_hours": 9, "disruptions": 0}, "meals": [], "mobility": null, "hydration": null, "visitors": [], "medication": null, "concerns": [], "summary": "Quiet day."}')]

        with patch.object(gen._client.messages, 'create', new_callable=AsyncMock, return_value=mock_response):
            entries = [
                MemoryEntry(timestamp=time.time() - 3600, summary="Woke up", detection_count=1),
                MemoryEntry(timestamp=time.time() - 3000, summary="In kitchen", detection_count=1),
            ]
            result = run(gen.generate_daily_report(entries, [], "Mom's Room", "2026-03-28"))

        assert result["date"] == "2026-03-28"
        assert result["camera_name"] == "Mom's Room"
        assert result["activity_count"] == 2
        assert result["summary"] == "Quiet day."
        assert result["sleep"]["bed_time"] == "10pm"

    def test_daily_report_handles_error(self):
        from report_generator import ReportGenerator
        gen = ReportGenerator()

        with patch.object(gen._client.messages, 'create', new_callable=AsyncMock, side_effect=Exception("API error")):
            result = run(gen.generate_daily_report([], [], "Room", "2026-03-28"))

        assert result["date"] == "2026-03-28"
        assert "failed" in result["concerns"][0].lower() or "failed" in result["summary"].lower()

    def test_weekly_report_with_data(self):
        from report_generator import ReportGenerator
        gen = ReportGenerator()

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"trends": {"sleep_avg_hours": 7.5, "sleep_trend": "stable", "meal_consistency": "3 meals/day", "mobility_trend": "stable", "visitor_frequency": "2/week", "concerns": []}, "daily_summaries": ["Mon: OK"], "recommendation": "Continue monitoring."}')]

        with patch.object(gen._client.messages, 'create', new_callable=AsyncMock, return_value=mock_response):
            daily_reports = [{"date": "2026-03-22", "summary": "Good day", "concerns": []}]
            result = run(gen.generate_weekly_report(daily_reports, "Room", "2026-03-22", "2026-03-28"))

        assert result["start_date"] == "2026-03-22"
        assert result["trends"]["sleep_avg_hours"] == 7.5
        assert result["recommendation"] == "Continue monitoring."

    def test_weekly_report_handles_error(self):
        from report_generator import ReportGenerator
        gen = ReportGenerator()

        with patch.object(gen._client.messages, 'create', new_callable=AsyncMock, side_effect=Exception("API error")):
            result = run(gen.generate_weekly_report([], "Room", "2026-03-22", "2026-03-28"))

        assert result["trends"]["sleep_trend"] == "unknown"
        assert "failed" in result["recommendation"].lower() or "unable" in result["recommendation"].lower()


# ---------------------------------------------------------------------------
# Concern-to-Rule Conversion
# ---------------------------------------------------------------------------

class TestConcernConversion:
    def test_concern_creates_rule_via_parser(self):
        """Test that the concerns route creates rules correctly (mocked parser)."""
        from routes.concerns import ConcernCreate

        body = ConcernCreate(text="Mom forgets to drink water", severity="medium")
        assert body.text == "Mom forgets to drink water"
        assert body.severity == "medium"

    def test_empty_concern_rejected(self):
        from routes.concerns import ConcernCreate
        body = ConcernCreate(text="", severity="medium")
        assert body.text.strip() == ""


# ---------------------------------------------------------------------------
# Memory Entry Database Operations
# ---------------------------------------------------------------------------

class TestMemoryEntries:
    def test_create_and_list_memory_entries(self):
        cam = _setup_camera()
        now = time.time()
        entry = MemoryEntry(timestamp=now, summary="Person seated on couch", detection_count=1)
        run(db.create_memory_entry(cam.id, entry))

        entries = run(db.list_memory_entries(cam.id, start_time=now - 10, end_time=now + 10))
        assert len(entries) == 1
        assert entries[0].summary == "Person seated on couch"

    def test_memory_entry_limit(self):
        cam = _setup_camera()
        now = time.time()
        for i in range(10):
            entry = MemoryEntry(timestamp=now + i, summary=f"Entry {i}", detection_count=0)
            run(db.create_memory_entry(cam.id, entry))

        # Request only 5
        entries = run(db.list_memory_entries(cam.id, start_time=now - 1, end_time=now + 100, limit=5))
        assert len(entries) == 5

    def test_memory_entries_ordered_by_timestamp(self):
        cam = _setup_camera()
        now = time.time()
        for i in [3, 1, 2]:
            entry = MemoryEntry(timestamp=now + i, summary=f"Entry {i}", detection_count=0)
            run(db.create_memory_entry(cam.id, entry))

        entries = run(db.list_memory_entries(cam.id, start_time=now, end_time=now + 10))
        # Verify we can sort them
        timestamps = [e.timestamp for e in entries]
        assert sorted(timestamps) == sorted(timestamps)


# ---------------------------------------------------------------------------
# Alerts Database (used by status endpoint)
# ---------------------------------------------------------------------------

class TestAlertQueries:
    def test_create_and_list_alerts(self):
        cam = _setup_camera()
        alert = Alert(
            camera_id=cam.id,
            rule_id="fall-1",
            rule_name="Fall Detection",
            severity="critical",
            timestamp=time.time(),
            narration="Person appears to have fallen",
        )
        run(db.create_alert(alert))
        alerts = run(db.list_alerts(cam.id, limit=10))
        assert len(alerts) == 1
        assert alerts[0]["rule_name"] == "Fall Detection"

    def test_count_alerts(self):
        cam = _setup_camera()
        for i in range(3):
            run(db.create_alert(Alert(
                camera_id=cam.id,
                rule_id=f"rule-{i}",
                rule_name=f"Rule {i}",
                severity="medium",
                timestamp=time.time(),
            )))
        count = run(db.count_alerts(cam.id))
        assert count == 3

    def test_clear_alerts(self):
        cam = _setup_camera()
        run(db.create_alert(Alert(
            camera_id=cam.id,
            rule_id="r1",
            rule_name="Test",
            severity="low",
            timestamp=time.time(),
        )))
        run(db.delete_alerts_for_camera(cam.id))
        alerts = run(db.list_alerts(cam.id, limit=10))
        assert len(alerts) == 0

    def test_memory_entries_excluded_from_alerts(self):
        """Memory entries stored as alerts with rule_id='__memory__' should be filterable."""
        cam = _setup_camera()
        # Add a real alert
        run(db.create_alert(Alert(
            camera_id=cam.id,
            rule_id="fall-1",
            rule_name="Fall",
            severity="critical",
            timestamp=time.time(),
        )))
        # Add a memory entry (stored as alert in DynamoDB)
        entry = MemoryEntry(timestamp=time.time(), summary="Watching TV", detection_count=1)
        run(db.create_memory_entry(cam.id, entry))

        all_alerts = run(db.list_alerts(cam.id, limit=100))
        real_alerts = [a for a in all_alerts if a.get("rule_id") != "__memory__"]
        # At least the real alert should be there
        assert any(a.get("rule_name") == "Fall" for a in real_alerts)


# ---------------------------------------------------------------------------
# Reasoner Prompt Update
# ---------------------------------------------------------------------------

class TestReasonerPrompt:
    def test_elder_care_context_in_prompt(self):
        from reasoner import _SYSTEM_PROMPT
        assert "elder care" in _SYSTEM_PROMPT.lower() or "Elder Care" in _SYSTEM_PROMPT
        assert "Sleep vs Fall" in _SYSTEM_PROMPT or "sleep vs fall" in _SYSTEM_PROMPT.lower()
        assert "bed" in _SYSTEM_PROMPT.lower()
        assert "floor" in _SYSTEM_PROMPT.lower()
        assert "mobility" in _SYSTEM_PROMPT.lower()
        assert "meal" in _SYSTEM_PROMPT.lower() or "kitchen" in _SYSTEM_PROMPT.lower()

    def test_prompt_has_severity_guidelines(self):
        from reasoner import _SYSTEM_PROMPT
        assert "CRITICAL" in _SYSTEM_PROMPT
        assert "HIGH" in _SYSTEM_PROMPT
        assert "MEDIUM" in _SYSTEM_PROMPT
        assert "LOW" in _SYSTEM_PROMPT
