from __future__ import annotations

import logging
import time
from typing import Any

from models import Alert, BBox, Detection, Rule, Zone

log = logging.getLogger("watchtower.rule_engine")

# Minimum seconds between repeated alerts for the same rule
_COOLDOWN = 15.0


def _bbox_in_zone(bbox: BBox, zone: Zone) -> bool:
    """Check if the center of a bounding box falls within a zone."""
    cx = bbox.x + bbox.width / 2
    cy = bbox.y + bbox.height / 2
    return (
        zone.x <= cx <= zone.x + zone.width
        and zone.y <= cy <= zone.y + zone.height
    )


def _bbox_height_ratio(bbox: BBox) -> float:
    """Return height as fraction of frame (0-1). Proxy for person size."""
    return bbox.height / 100.0


def _estimate_pose_state(detection: Detection) -> str:
    """Estimate pose from keypoints or bbox fallback: standing, sitting, lying, crouching."""

    # Fallback: use bounding box aspect ratio when no keypoints (e.g., Lambda without MediaPipe)
    if not detection.pose:
        bbox = detection.bbox
        if bbox.width > 0 and bbox.height > 0:
            aspect = bbox.width / bbox.height
            # Person wider than tall → likely lying down
            if aspect > 1.2:
                return "lying"
            # Very squat → sitting or crouching
            if aspect > 0.85:
                return "sitting"
            return "standing"
        return "unknown"

    kp = {p.name: p for p in detection.pose}

    left_hip = kp.get("left_hip")
    right_hip = kp.get("right_hip")
    left_shoulder = kp.get("left_shoulder")
    right_shoulder = kp.get("right_shoulder")
    left_knee = kp.get("left_knee")
    right_knee = kp.get("right_knee")

    if not all([left_hip, right_hip, left_shoulder, right_shoulder]):
        return "unknown"

    hip_y = (left_hip.y + right_hip.y) / 2
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    torso_height = abs(hip_y - shoulder_y)

    hip_x = (left_hip.x + right_hip.x) / 2
    shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    torso_width = abs(hip_x - shoulder_x)

    # Lying: torso is more horizontal than vertical
    if torso_width > torso_height * 1.5:
        return "lying"

    # If knees are available, check crouch
    if left_knee and right_knee:
        knee_y = (left_knee.y + right_knee.y) / 2
        # Crouching: hips close to knees vertically
        if abs(hip_y - knee_y) < torso_height * 0.5:
            return "crouching"

    # Short torso relative to expected = sitting
    if torso_height < 8:  # very compressed torso in percentage terms
        return "sitting"

    return "standing"


class RuleEngine:
    def __init__(self) -> None:
        # Track when each rule last fired (rule_id -> timestamp)
        self._last_fired: dict[str, float] = {}
        # Track duration conditions (rule_id -> first_true_timestamp)
        self._duration_tracking: dict[str, float] = {}

    def evaluate(
        self,
        rules: list[Rule],
        zones: list[Zone],
        detections: list[Detection],
        now: float,
    ) -> list[Alert]:
        fired: list[Alert] = []

        for rule in rules:
            if not rule.enabled:
                continue

            # Cooldown check
            last = self._last_fired.get(rule.id, 0.0)
            if now - last < _COOLDOWN:
                continue

            matched_detections: list[Detection] = []

            # Separate duration from other conditions so duration tracking
            # only resets when non-duration conditions fail.
            non_duration = [c for c in rule.conditions if c.type != "duration"]
            duration_conds = [c for c in rule.conditions if c.type == "duration"]

            non_duration_met = True
            for condition in non_duration:
                met, matches = self._check_condition(
                    condition.type, condition.params, zones, detections, rule.id, now,
                )
                if not met:
                    non_duration_met = False
                    break
                matched_detections.extend(matches)

            if not non_duration_met:
                # Non-duration conditions failed: reset duration tracking
                self._duration_tracking.pop(rule.id, None)
                continue

            # Non-duration conditions passed. Now check duration if any.
            duration_met = True
            for condition in duration_conds:
                met, _ = self._check_condition(
                    condition.type, condition.params, zones, detections, rule.id, now,
                )
                if not met:
                    duration_met = False
                    break

            if duration_met and rule.conditions:
                self._last_fired[rule.id] = now
                self._duration_tracking.pop(rule.id, None)

                fired.append(Alert(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    severity=rule.severity,
                    timestamp=now,
                    detections=matched_detections,
                ))

        return fired

    def _check_condition(
        self,
        ctype: str,
        params: dict[str, Any],
        zones: list[Zone],
        detections: list[Detection],
        rule_id: str,
        now: float,
    ) -> tuple[bool, list[Detection]]:
        checker = _CONDITION_CHECKERS.get(ctype)
        if checker is None:
            log.warning("Unknown condition type: %s", ctype)
            return False, []
        return checker(self, params, zones, detections, rule_id, now)

    # --- Individual condition checkers ---

    def _check_object_present(
        self, params: dict, zones: list[Zone], detections: list[Detection],
        rule_id: str, now: float,
    ) -> tuple[bool, list[Detection]]:
        target = params.get("class", "person")
        matches = [d for d in detections if d.class_name == target]
        return bool(matches), matches

    def _check_object_absent(
        self, params: dict, zones: list[Zone], detections: list[Detection],
        rule_id: str, now: float,
    ) -> tuple[bool, list[Detection]]:
        target = params.get("class", "person")
        present = any(d.class_name == target for d in detections)
        return not present, []

    def _check_object_in_zone(
        self, params: dict, zones: list[Zone], detections: list[Detection],
        rule_id: str, now: float,
    ) -> tuple[bool, list[Detection]]:
        target_class = params.get("class", "person")
        zone_name = params.get("zone", "")

        zone = next((z for z in zones if z.name.lower() == zone_name.lower()), None)
        if zone is None:
            return False, []

        matches = [
            d for d in detections
            if d.class_name == target_class and _bbox_in_zone(d.bbox, zone)
        ]
        return bool(matches), matches

    def _check_object_not_in_zone(
        self, params: dict, zones: list[Zone], detections: list[Detection],
        rule_id: str, now: float,
    ) -> tuple[bool, list[Detection]]:
        target_class = params.get("class", "person")
        zone_name = params.get("zone", "")

        zone = next((z for z in zones if z.name.lower() == zone_name.lower()), None)
        if zone is None:
            return True, []

        in_zone = any(
            d.class_name == target_class and _bbox_in_zone(d.bbox, zone)
            for d in detections
        )
        return not in_zone, []

    def _check_person_size(
        self, params: dict, zones: list[Zone], detections: list[Detection],
        rule_id: str, now: float,
    ) -> tuple[bool, list[Detection]]:
        size = params.get("size", "small")  # small = child, large = adult
        threshold = params.get("threshold", 0.3)  # 30% of frame height

        people = [d for d in detections if d.class_name == "person"]
        if size == "small":
            matches = [d for d in people if _bbox_height_ratio(d.bbox) < threshold]
        else:
            matches = [d for d in people if _bbox_height_ratio(d.bbox) >= threshold]
        return bool(matches), matches

    def _check_person_pose(
        self, params: dict, zones: list[Zone], detections: list[Detection],
        rule_id: str, now: float,
    ) -> tuple[bool, list[Detection]]:
        target_pose = params.get("pose", "standing")
        people = [d for d in detections if d.class_name == "person"]
        matches = [d for d in people if _estimate_pose_state(d) == target_pose]
        return bool(matches), matches

    def _check_count(
        self, params: dict, zones: list[Zone], detections: list[Detection],
        rule_id: str, now: float,
    ) -> tuple[bool, list[Detection]]:
        target_class = params.get("class", "person")
        operator = params.get("operator", "gte")  # gte, lte, eq, gt, lt
        value = params.get("value", 1)

        matching = [d for d in detections if d.class_name == target_class]
        count = len(matching)

        ops = {
            "gte": count >= value,
            "lte": count <= value,
            "eq": count == value,
            "gt": count > value,
            "lt": count < value,
        }
        return ops.get(operator, False), matching

    def _check_duration(
        self, params: dict, zones: list[Zone], detections: list[Detection],
        rule_id: str, now: float,
    ) -> tuple[bool, list[Detection]]:
        seconds = params.get("seconds", 5)

        if rule_id not in self._duration_tracking:
            self._duration_tracking[rule_id] = now
            return False, []

        elapsed = now - self._duration_tracking[rule_id]
        return elapsed >= seconds, []

    def _check_time_window(
        self, params: dict, zones: list[Zone], detections: list[Detection],
        rule_id: str, now: float,
    ) -> tuple[bool, list[Detection]]:
        start_hour = params.get("start_hour", 0)
        end_hour = params.get("end_hour", 24)

        import datetime
        current_hour = datetime.datetime.now().hour
        if start_hour <= end_hour:
            in_window = start_hour <= current_hour < end_hour
        else:
            in_window = current_hour >= start_hour or current_hour < end_hour
        return in_window, []


_CONDITION_CHECKERS = {
    "object_present": RuleEngine._check_object_present,
    "object_absent": RuleEngine._check_object_absent,
    "object_in_zone": RuleEngine._check_object_in_zone,
    "object_not_in_zone": RuleEngine._check_object_not_in_zone,
    "person_size": RuleEngine._check_person_size,
    "person_pose": RuleEngine._check_person_pose,
    "count": RuleEngine._check_count,
    "duration": RuleEngine._check_duration,
    "time_window": RuleEngine._check_time_window,
}
