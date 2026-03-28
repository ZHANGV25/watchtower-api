from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass

import anthropic
import cv2
import numpy as np

from models import Alert, Detection

log = logging.getLogger("watchtower.narrator")

_BEDROCK_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

_NARRATION_PROMPT = """\
You are a live scene narrator for WatchTower, a camera monitoring system. \
Describe what is happening in this frame in 1-2 natural sentences. \
Be specific about people, objects, positions, and activities. \
Speak as if narrating to someone who cannot see the screen. \
Be calm and conversational, not clinical. Keep it under 30 words."""

_SYSTEM_PROMPT = """You are a verification gate for WatchTower, a camera monitoring system.

A detection rule has fired based on YOLO object detection. Your job is to look at the camera frame and verify whether the alert is a true positive or a false positive.

Respond with ONLY valid JSON (no markdown, no explanation):
{"confirmed": true} or {"confirmed": false}

Only add a "note" field if the situation is genuinely ambiguous or noteworthy:
{"confirmed": true, "note": "Two people near the restricted zone, one may be entering"}

Rules:
- confirmed=true: The scene clearly matches what the rule describes
- confirmed=false: YOLO misidentified something, or the scene does not match the rule
- Keep notes under 20 words. Most responses should have no note at all.
- When in doubt, confirm. False negatives are worse than false positives."""


@dataclass
class VerificationResult:
    confirmed: bool
    note: str


class Narrator:
    def __init__(self) -> None:
        self._client = anthropic.AsyncAnthropicBedrock(
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
        )

    async def verify(self, frame: np.ndarray, alert: Alert) -> VerificationResult:
        """Verify whether an alert is a true positive. Returns confirmation + optional note."""
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return VerificationResult(confirmed=True, note="")

        b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        context = (
            f"Rule: {alert.rule_name}\n"
            f"Severity: {alert.severity}\n"
            f"YOLO detections: {', '.join(d.class_name for d in alert.detections)}"
        )

        try:
            response = await self._client.messages.create(
                model=_BEDROCK_MODEL,
                max_tokens=100,
                system=_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": context,
                        },
                    ],
                }],
            )

            raw = response.content[0].text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                # Remove optional language tag (e.g. "json\n")
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)
            return VerificationResult(
                confirmed=parsed.get("confirmed", True),
                note=parsed.get("note", ""),
            )

        except Exception as e:
            log.error("Verification failed: %s", e)
            # Default to confirmed on error (don't suppress real alerts)
            return VerificationResult(confirmed=True, note="")

    async def narrate_scene(
        self, frame: np.ndarray, detections: list[Detection] | None = None
    ) -> str:
        """Generate a live narration of the current scene (Block 5)."""
        if frame.size == 0:
            return ""
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not ok:
            return ""

        b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        det_context = ""
        if detections:
            det_context = f"\nDetected: {', '.join(d.class_name for d in detections[:8])}"

        try:
            response = await self._client.messages.create(
                model=_BEDROCK_MODEL,
                max_tokens=80,
                system=_NARRATION_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Narrate this scene.{det_context}",
                        },
                    ],
                }],
            )
            return response.content[0].text.strip()
        except Exception as e:
            log.error("Narration failed: %s", e)
            return ""

    async def compare_anomaly(
        self, baseline_frame: np.ndarray, current_frame: np.ndarray, score: float
    ) -> str:
        """Compare current frame against baseline and describe what changed."""
        if baseline_frame.size == 0 or current_frame.size == 0:
            return "Anomalous activity detected."

        ok1, buf1 = cv2.imencode(".jpg", baseline_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        ok2, buf2 = cv2.imencode(".jpg", current_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if not ok1 or not ok2:
            return "Anomalous activity detected."

        b64_baseline = base64.b64encode(buf1.tobytes()).decode("ascii")
        b64_current = base64.b64encode(buf2.tobytes()).decode("ascii")

        try:
            response = await self._client.messages.create(
                model=_BEDROCK_MODEL,
                max_tokens=120,
                system=(
                    "You are an anomaly detection system. The first image is the BASELINE "
                    "(what the scene normally looks like). The second image is the CURRENT "
                    "frame that triggered an anomaly alert. Describe specifically what "
                    "changed or what is different. Focus on new objects, missing objects, "
                    "people appearing/disappearing, or significant changes in the scene. "
                    "Be concise (1-2 sentences). Do NOT describe the whole scene."
                ),
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64_baseline,
                            },
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64_current,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"Anomaly score: {score:.0%}. What changed between baseline and current?",
                        },
                    ],
                }],
            )
            return response.content[0].text.strip()
        except Exception as e:
            log.error("Anomaly comparison failed: %s", e)
            return "Anomalous activity detected."
