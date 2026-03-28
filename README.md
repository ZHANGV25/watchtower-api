# WatchTower API

Backend for WatchTower, an AI-powered real-time camera monitoring system. Describe what to watch for in plain English and WatchTower handles the rest.

Built for YHack 2026.

## What It Does

Captures live camera video, runs YOLO object detection + MediaPipe pose estimation every frame, evaluates user-defined rules, and fires alerts verified by a vision LLM. Includes six intelligence layers:

- **Auto-Bootstrap** - Analyzes the scene on startup, suggests zones and monitoring rules automatically
- **Rule Engine** - Natural language rules compiled to structured conditions (object presence, zones, pose, duration, count, time windows)
- **Multi-Frame Reasoning** - LLM analyzes 4 frames every 10 seconds to understand sequences, intent, and context
- **Scene Memory + Investigation** - Logs scene summaries every 30 seconds, answers natural language questions about the past
- **Live Narration** - Continuous scene description for accessibility or ambient awareness
- **Anomaly Detection** - Learns what "normal" looks like, alerts when something changes without needing explicit rules
- **Agentic Actions** - Alerts trigger TTS (ElevenLabs), browser sounds, and webhooks

## Tech Stack

- **Framework:** FastAPI + WebSockets
- **Computer Vision:** YOLO v8n (ultralytics) + MediaPipe Pose
- **LLM:** Claude via AWS Bedrock (Sonnet for reasoning/parsing, Haiku for verification/narration)
- **Real-time:** All communication via WebSocket, async non-blocking LLM calls

## Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Variables

- `AWS_REGION` - AWS region for Bedrock (default: `us-east-1`)
- `ELEVENLABS_API_KEY` - Optional, for TTS alerts (falls back to browser speech)
- `ELEVENLABS_VOICE_ID` - Optional, ElevenLabs voice
- `WATCHTOWER_WEBHOOK_URL` - Optional, POST alerts to this URL
- `WATCHTOWER_NO_CAMERA` - Set to `1` to disable camera (testing)
- `WATCHTOWER_SEG` - Set to `1` for segmentation masks instead of bounding boxes

## Tests

```bash
python -m pytest tests/ -v --ignore=tests/test_websocket_integration.py --ignore=tests/test_rule_parser_integration.py --ignore=tests/test_plan_generator.py
```

142 tests covering unit tests for all modules and integration tests for all 21 WebSocket handlers.

## Architecture

```
Camera (24 fps)
  |
  v
YOLO v8n + MediaPipe Pose --> Detections
  |
  +--> Rule Engine (boolean conditions per frame) --> Alerts
  |      |
  |      +--> LLM Verification (Haiku) --> Action Engine (TTS/webhook/sound)
  |
  +--> Reasoning Loop (Sonnet, every 10s, multi-frame)
  |
  +--> Memory Loop (Haiku, every 30s, scene summaries)
  |
  +--> Narration Loop (Haiku, every 8s, live commentary)
  |
  +--> Anomaly Detection (frame features, every 2s)
  |
  +--> Replay Buffer (30 min circular, 2 fps)
```

## WebSocket Protocol

All communication via `ws://localhost:8000/ws`. Messages are JSON: `{"type": "...", "payload": {...}}`.

21 message types: `add_rule`, `toggle_rule`, `delete_rule`, `update_rule`, `update_zones`, `auto_zones`, `generate_plan`, `apply_plan`, `approve_bootstrap`, `dismiss_bootstrap`, `toggle_reasoning`, `toggle_narration`, `toggle_anomaly`, `set_anomaly_threshold`, `update_actions`, `ask`, `get_replay`, `get_replay_timestamps`, `get_frame_at`, `clear_alerts`, `clear_rules`, `reset_all`.

## Project Structure

```
main.py              # FastAPI app, WebSocket handlers, background loops
detector.py          # YOLO + MediaPipe wrapper
rule_engine.py       # Per-frame condition evaluation
rule_parser.py       # NL -> JSON rule compiler (Sonnet)
plan_generator.py    # Multi-rule scenario generator (Sonnet)
scene_analyzer.py    # Auto-bootstrap scene analysis (Sonnet)
reasoner.py          # Multi-frame reasoning loop (Sonnet)
narrator.py          # Alert verification + live narration + anomaly comparison (Haiku)
memory.py            # Scene memory log + investigation (Haiku/Sonnet)
actions.py           # TTS, webhooks, browser sounds
anomaly.py           # Frame feature extraction + anomaly scoring
replay_buffer.py     # 30-min circular frame buffer
zone_generator.py    # Auto-zone detection from frame (Sonnet)
models.py            # Pydantic data models
mask_utils.py        # Segmentation polygon extraction
```
