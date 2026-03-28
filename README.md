# WatchTower API

Backend for WatchTower, an AI-powered elder care monitoring system. A $50 camera device that watches over elderly loved ones and alerts caregivers when something goes wrong.

Built for YHack 2026.

## What It Does

Processes video clips from in-home cameras to detect falls, track daily activity patterns, monitor sleep schedules, and alert caregivers to potential emergencies. Caregivers can ask natural language questions like "What did Mom do today?" and get answers from the scene memory.

## Elder Care Features

### Preset Monitoring (always on)
- **Fall detection** — Person lying on floor outside of bed area, distinguished from sleeping by location + time of day
- **Inactivity alert** — No movement detected for 3+ hours during daytime
- **Night wandering** — Person detected moving around between 11pm-5am
- **Visitor detection** — Multiple people in frame (caregiver arrival, unexpected visitor)
- **Sleep tracking** — Time to bed, wake time, nighttime disruptions
- **Medication reminders** — Alerts caregiver if no activity near medication area by scheduled time
- **Emergency detection** — Person on floor + no movement for extended period

### Custom Concerns (user-added)
Caregivers add concerns in plain English: "Mom forgets to drink water" or "Dad shouldn't be climbing stairs." The AI translates these into monitoring rules automatically.

### Activity Timeline
Scene memory logs activity every 30 seconds. The timeline shows: "7:15 AM — Woke up, moved to kitchen" / "7:30 AM — Seated at table (breakfast)" / "10:30 AM — On couch, watching TV."

### Investigation
Ask questions in natural language: "What did she do this morning?" / "When did she last eat?" / "Did anyone visit today?" The AI answers from its scene memory.

### Daily Reports (ADL Tracking)
Generates structured Activities of Daily Living reports for doctors and caregivers:
```
Daily Report: Mom — March 28, 2026
Sleep: 10:45pm - 6:30am (7h 45m, 1 disruption at 2:15am)
Meals: Breakfast 7:15am (25 min), Lunch 12:30pm (20 min), Dinner skipped
Mobility: 14 room transitions, mostly kitchen↔living room
Hydration: 3 cup detections (below recommended)
Visitors: 1 visit at 3pm (45 min)
Concerns: Dinner skipped, below-average hydration
```
The LLM reasoning engine compiles activity timeline entries into structured daily/weekly summaries. Exportable for clinical review.

### Continuous Activity Logging
Every 30 seconds, the system logs what the person is doing — mobility, posture, room location, objects in use. This creates a complete record of daily activity patterns that doctors can review for:
- Declining mobility trends
- Sleep quality deterioration
- Meal/hydration frequency changes
- Social isolation (fewer visitors)
- Behavioral changes indicating cognitive decline

## Architecture

```
Camera (Pi + USB webcam in each room)
  → Motion detected → Record clip → Upload to S3
  → S3 trigger → Lambda (YOLO + rules + LLM reasoning)
  → DynamoDB (alerts, activity log, sleep data)
  → Ntfy push notification to caregiver's phone

Caregiver opens dashboard:
  → REST API (Lambda + API Gateway) → DynamoDB
  → WebRTC live check-in (P2P to camera)
```

Fully serverless. No EC2. Zero idle cost.

## Tech Stack

- **Framework:** FastAPI + Lambda (via Mangum)
- **Computer Vision:** YOLO v8n (ultralytics)
- **LLM:** Claude via AWS Bedrock (Sonnet for reasoning, Haiku for narration)
- **Database:** DynamoDB (cameras/rooms, rules, alerts, activity log, users)
- **Storage:** S3 (video clips, alert frames)
- **Notifications:** Ntfy (push to phone)
- **Live View:** WebRTC (P2P, no server relay)

## API Endpoints

### Auth
- `POST /api/auth/register` — Create account
- `POST /api/auth/login` — Sign in, get JWT
- `GET /api/auth/me` — Current user

### Cameras (Rooms)
- `GET /api/cameras` — List all rooms/cameras
- `POST /api/cameras` — Register a new room
- `GET /api/cameras/{id}` — Room details + alert count
- `PUT /api/cameras/{id}` — Update room name/location
- `DELETE /api/cameras/{id}` — Remove room

### Rules (Monitoring Concerns)
- `GET /api/cameras/{id}/rules` — List monitoring rules for a room
- `DELETE /api/cameras/{id}/rules/{rule_id}` — Remove a concern
- `PATCH /api/cameras/{id}/rules/{rule_id}/toggle` — Enable/disable

### Alerts
- `GET /api/cameras/{id}/alerts` — Alert history (paginated)
- `GET /api/alerts/{id}` — Single alert with frame
- `DELETE /api/cameras/{id}/alerts` — Clear alerts

### Clip Processing
- `POST /api/clips/process` — Process clip from S3 (called by S3 trigger)
- `POST /api/clips/upload` — Direct clip upload

## Setup

### Local development
```bash
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Lambda deployment
```bash
# Build and push container images
docker buildx build -f Dockerfile.api -t <ECR_URI>/watchtower-api:latest --platform linux/amd64 --provenance=false --push .
docker buildx build -f Dockerfile.clip -t <ECR_URI>/watchtower-clip-processor:latest --platform linux/amd64 --provenance=false --push .
```

### Environment Variables
- `WATCHTOWER_DB_BACKEND` — `sqlite` (local) or `dynamodb` (Lambda)
- `WATCHTOWER_SECRET` — JWT signing key
- `WATCHTOWER_NTFY_TOPIC` — Push notification topic
- `WATCHTOWER_S3_BUCKET` — S3 bucket for clips/frames
- `WATCHTOWER_STORAGE` — `local` or `s3`
- `AWS_REGION` — AWS region (default: us-east-1)

## Tests

```bash
python -m pytest tests/ -v --ignore=tests/test_websocket_integration.py --ignore=tests/test_rule_parser_integration.py --ignore=tests/test_plan_generator.py
```

129 tests covering database CRUD, auth, WebSocket handlers, rule engine, anomaly detection, scene analysis, memory, and actions.

## Project Structure

```
main.py              # FastAPI app, multi-camera sessions, processing loops
database.py          # SQLite async CRUD
database_dynamo.py   # DynamoDB CRUD (Lambda mode)
db.py                # Backend selector shim
camera_manager.py    # Per-camera/room session management
detector.py          # YOLO v8n + optional MediaPipe
rule_engine.py       # Per-frame condition evaluation
rule_parser.py       # Natural language → monitoring rules (Claude Sonnet)
narrator.py          # Alert verification + scene narration (Claude Haiku)
reasoner.py          # Multi-frame reasoning (sleep vs fall, activity patterns)
memory.py            # Activity timeline + investigation Q&A
anomaly.py           # Baseline learning + anomaly detection
actions.py           # Push notifications, TTS, webhooks
scene_analyzer.py    # Room analysis on first frame
zone_generator.py    # Auto-detect room areas (bed, door, kitchen, etc.)
storage.py           # Frame storage (local filesystem or S3)
auth.py              # JWT tokens + password hashing
middleware.py        # Auth dependencies
models.py            # All data models
routes/              # REST API endpoints
lambda_api.py        # Lambda handler for REST API
lambda_clip.py       # Lambda handler for clip processing
```
