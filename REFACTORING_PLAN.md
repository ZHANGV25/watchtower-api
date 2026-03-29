# WatchTower API Backend - Refactoring Plan

## Current State

The API currently:
- Manages WebSocket connections for real-time frame streaming
- Maintains in-memory session state per camera
- Processes frames in real-time processing loops
- Uses SQLite for persistence
- Has complex camera session management

**Problems**:
- WebSocket complexity and connection management
- In-memory state doesn't work with Lambda
- Not horizontally scalable
- Real-time processing loop ties up resources

## Target State

**Serverless Lambda + PostgreSQL RDS**:
- Event-driven segment processing (S3 triggers)
- Stateless Lambda functions
- REST API only (no WebSockets)
- PostgreSQL for all persistence
- RDS Proxy for connection pooling

## Repository Structure Changes

### DELETE (Remove Completely)
- [ ] `camera_manager.py` - In-memory session state management
- [ ] All WebSocket endpoint code in `main.py`
- [ ] `database.py` - SQLite implementation (replace with PostgreSQL)

### CREATE
- [ ] `database_postgres.py` - PostgreSQL adapter with asyncpg
- [ ] `lambda_segment.py` - Lambda handler for HLS segment processing
- [ ] `lambda_api.py` - Lambda handler for REST API (already exists, update)
- [ ] `Dockerfile` - Container image for Lambda
- [ ] `db_migration.sql` - PostgreSQL schema

### MODIFY
- [ ] `main.py` - Remove WebSockets, simplify
- [ ] `db.py` - Update selector for PostgreSQL
- [ ] `routes/cameras.py` - Add stream-url endpoint
- [ ] `routes/clips.py` - Remove, replaced by segment processing
- [ ] `requirements.txt` - Update dependencies

## Detailed Refactoring Steps

### Step 1: Create PostgreSQL Database Layer

**New file**: `database_postgres.py`

```python
"""PostgreSQL database adapter using asyncpg."""

import asyncpg
import os
import json
import time
import uuid
from typing import Optional, List, Dict, Any
import boto3

# Global connection pool (reused across Lambda invocations)
_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Get or create connection pool."""
    global _pool

    if _pool is None:
        # Get database credentials from Secrets Manager
        secrets_client = boto3.client('secretsmanager')
        secret_arn = os.environ['DB_SECRET_ARN']

        secret_value = secrets_client.get_secret_value(SecretId=secret_arn)
        secret = json.loads(secret_value['SecretString'])

        _pool = await asyncpg.create_pool(
            host=os.environ['DB_HOST'],
            port=int(os.environ.get('DB_PORT', 5432)),
            database=os.environ['DB_NAME'],
            user=secret['username'],
            password=secret['password'],
            min_size=1,
            max_size=5,  # Per Lambda container
            command_timeout=60,
        )

    return _pool


async def init_db():
    """Initialize database schema."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        # Cameras table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                location TEXT,
                status TEXT DEFAULT 'offline',
                last_seen DOUBLE PRECISION,
                stream_url TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        # Stream segments table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS stream_segments (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
                segment_number INTEGER NOT NULL,
                timestamp DOUBLE PRECISION NOT NULL,
                duration DOUBLE PRECISION NOT NULL,
                s3_key TEXT NOT NULL,
                processed BOOLEAN DEFAULT FALSE,
                alert_count INTEGER DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_segments_camera_timestamp
            ON stream_segments(camera_id, timestamp DESC)
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_segments_unprocessed
            ON stream_segments(camera_id, processed) WHERE processed = FALSE
        """)

        # Rules table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS rules (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                natural_language TEXT NOT NULL,
                conditions JSONB NOT NULL,
                severity TEXT NOT NULL,
                enabled BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        # Zones table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS zones (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                x DOUBLE PRECISION NOT NULL,
                y DOUBLE PRECISION NOT NULL,
                width DOUBLE PRECISION NOT NULL,
                height DOUBLE PRECISION NOT NULL,
                color TEXT DEFAULT '#00ffff',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        # Alerts table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
                rule_id UUID REFERENCES rules(id) ON DELETE SET NULL,
                rule_name TEXT NOT NULL,
                severity TEXT NOT NULL,
                timestamp DOUBLE PRECISION NOT NULL,
                frame_s3_key TEXT,
                narration TEXT,
                detections JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_camera_timestamp
            ON alerts(camera_id, timestamp DESC)
        """)

        # Memory entries table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_entries (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
                timestamp DOUBLE PRECISION NOT NULL,
                summary TEXT NOT NULL,
                detection_count INTEGER DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        # Users table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)


# CRUD operations

async def create_camera(camera: Dict[str, Any]) -> str:
    """Create new camera."""
    pool = await get_pool()
    camera_id = camera.get('id', str(uuid.uuid4()))

    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO cameras (id, name, location, status)
               VALUES ($1, $2, $3, $4)""",
            camera_id, camera['name'], camera.get('location'), 'offline'
        )

    return camera_id


async def get_camera(camera_id: str) -> Optional[Dict[str, Any]]:
    """Get camera by ID."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM cameras WHERE id = $1",
            camera_id
        )

    return dict(row) if row else None


async def list_cameras() -> List[Dict[str, Any]]:
    """List all cameras."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM cameras ORDER BY created_at DESC")

    return [dict(row) for row in rows]


async def create_stream_segment(segment: Dict[str, Any]) -> str:
    """Create stream segment record."""
    pool = await get_pool()
    segment_id = str(uuid.uuid4())

    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO stream_segments
               (id, camera_id, segment_number, timestamp, duration, s3_key, processed, alert_count)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
            segment_id,
            segment['camera_id'],
            segment.get('segment_number', 0),
            segment['timestamp'],
            segment.get('duration', 3.0),
            segment['s3_key'],
            segment.get('processed', False),
            segment.get('alert_count', 0)
        )

    return segment_id


async def create_alert(alert: Dict[str, Any]) -> str:
    """Create alert."""
    pool = await get_pool()
    alert_id = str(uuid.uuid4())

    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO alerts
               (id, camera_id, rule_id, rule_name, severity, timestamp, frame_s3_key, detections)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
            alert_id,
            alert['camera_id'],
            alert.get('rule_id'),
            alert['rule_name'],
            alert['severity'],
            alert['timestamp'],
            alert.get('frame_s3_key'),
            json.dumps(alert.get('detections', []))
        )

    return alert_id


async def update_alert_narration(alert_id: str, narration: str):
    """Update alert with narration."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE alerts SET narration = $1 WHERE id = $2",
            narration, alert_id
        )


async def get_alerts(
    camera_id: str,
    since: float = 0,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """Get alerts for a camera since timestamp."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT * FROM alerts
               WHERE camera_id = $1 AND timestamp > $2
               ORDER BY timestamp DESC
               LIMIT $3""",
            camera_id, since, limit
        )

    return [dict(row) for row in rows]


async def get_rules(camera_id: str) -> List[Dict[str, Any]]:
    """Get rules for a camera."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT * FROM rules
               WHERE camera_id = $1 AND enabled = TRUE
               ORDER BY created_at DESC""",
            camera_id
        )

    # Parse JSONB conditions back to Python
    rules = []
    for row in rows:
        rule = dict(row)
        rule['conditions'] = json.loads(rule['conditions']) if isinstance(rule['conditions'], str) else rule['conditions']
        rules.append(rule)

    return rules


async def get_zones(camera_id: str) -> List[Dict[str, Any]]:
    """Get zones for a camera."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM zones WHERE camera_id = $1",
            camera_id
        )

    return [dict(row) for row in rows]
```

### Step 2: Create Lambda Segment Processor

**New file**: `lambda_segment.py`

```python
"""Lambda function to process HLS segments from S3."""

import json
import os
import time
import tempfile
import cv2
import numpy as np
import boto3
from typing import List

# Import processing modules
from detector import Detector
from rule_engine import RuleEngine
from narrator import Narrator
import database_postgres as db


# Initialize singletons (cached across Lambda invocations)
detector = Detector()
rule_engine = RuleEngine()
narrator = Narrator()
s3_client = boto3.client('s3')


def extract_frames_from_segment(segment_bytes: bytes, sample_rate: float = 2.0) -> List[np.ndarray]:
    """Extract frames from MPEG-TS segment.

    Args:
        segment_bytes: Raw .ts file bytes
        sample_rate: Extract one frame every N seconds

    Returns:
        List of frames as numpy arrays
    """
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as tmp:
        tmp.write(segment_bytes)
        tmp_path = tmp.name

    try:
        # Open with OpenCV
        cap = cv2.VideoCapture(tmp_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        frame_interval = int(fps * sample_rate)

        frames = []
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames at specified rate
            if frame_num % frame_interval == 0:
                frames.append(frame)

            frame_num += 1

        cap.release()
        return frames

    finally:
        os.unlink(tmp_path)


async def process_segment(event, context):
    """Lambda handler for segment processing.

    Triggered by S3 PUT event when new .ts segment is uploaded.
    """
    # Parse S3 event
    record = event['Records'][0]
    bucket = record['s3']['bucket']['name']
    key = record['s3']['object']['key']  # e.g., "live/camera-123/segment_001.ts"

    print(f"Processing segment: s3://{bucket}/{key}")

    # Extract camera ID from S3 key
    parts = key.split('/')
    if len(parts) < 3 or parts[0] != 'live':
        print(f"Invalid S3 key format: {key}")
        return {'statusCode': 400, 'body': 'Invalid key'}

    camera_id = parts[1]
    segment_name = parts[2]

    # Download segment from S3
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        segment_bytes = obj['Body'].read()
    except Exception as e:
        print(f"Failed to download segment: {e}")
        return {'statusCode': 500, 'body': str(e)}

    # Extract frames (sample every 2 seconds)
    frames = extract_frames_from_segment(segment_bytes, sample_rate=2.0)
    print(f"Extracted {len(frames)} frames from segment")

    if not frames:
        print("No frames extracted, skipping")
        return {'statusCode': 200, 'body': 'No frames'}

    # Get rules and zones for this camera
    rules = await db.get_rules(camera_id)
    zones = await db.get_zones(camera_id)

    # Process each frame
    alert_count = 0

    for frame_idx, frame in enumerate(frames):
        # Run YOLO detection
        detections = detector.detect(frame)

        # Evaluate rules
        triggered_rules = rule_engine.evaluate(
            camera_id=camera_id,
            detections=detections,
            frame=frame,
            rules=rules,
            zones=zones
        )

        # Create alerts for triggered rules
        for rule in triggered_rules:
            alert_count += 1

            # Upload alert frame to S3
            alert_id = f"{camera_id}_{int(time.time())}_{frame_idx}"
            frame_key = f"alerts/{alert_id}.jpg"

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            s3_client.put_object(
                Bucket=bucket,
                Key=frame_key,
                Body=buffer.tobytes(),
                ContentType='image/jpeg'
            )

            # Create alert in database
            alert_db_id = await db.create_alert({
                'camera_id': camera_id,
                'rule_id': rule.get('id'),
                'rule_name': rule['name'],
                'severity': rule['severity'],
                'timestamp': time.time(),
                'frame_s3_key': frame_key,
                'detections': detections
            })

            # Generate narration asynchronously (non-blocking)
            try:
                narration = await narrator.narrate(frame, detections, rule)
                await db.update_alert_narration(alert_db_id, narration)
            except Exception as e:
                print(f"Narration failed: {e}")

    # Save segment metadata
    await db.create_stream_segment({
        'camera_id': camera_id,
        's3_key': key,
        'timestamp': time.time(),
        'processed': True,
        'alert_count': alert_count
    })

    print(f"Processed segment with {alert_count} alerts")

    return {
        'statusCode': 200,
        'body': json.dumps({
            'processed': True,
            'frames': len(frames),
            'alerts': alert_count
        })
    }


def handler(event, context):
    """Synchronous Lambda handler (wraps async function)."""
    import asyncio
    return asyncio.run(process_segment(event, context))
```

### Step 3: Update Main API (Remove WebSockets)

**Modify**: `main.py`

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import database_postgres as db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    await db.init_db()
    yield


app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include route modules
from routes import cameras, alerts, rules, zones, auth

app.include_router(cameras.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(rules.router, prefix="/api")
app.include_router(zones.router, prefix="/api")
app.include_router(auth.router, prefix="/api/auth")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# DELETE: All WebSocket endpoint code
# DELETE: camera_processing_loop()
# DELETE: broadcast() function
# DELETE: frontend_clients tracking
```

### Step 4: Add Stream URL Endpoint

**Modify**: `routes/cameras.py`

```python
import boto3
import os
from fastapi import APIRouter, HTTPException
import database_postgres as db

router = APIRouter()
s3_client = boto3.client('s3')


@router.get("/cameras/{camera_id}/stream-url")
async def get_stream_url(camera_id: str):
    """Get presigned URL for HLS playlist."""
    bucket = os.environ['S3_BUCKET']
    playlist_key = f'live/{camera_id}/playlist.m3u8'

    # Check if camera exists
    camera = await db.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Check if stream is active (playlist exists in S3)
    try:
        s3_client.head_object(Bucket=bucket, Key=playlist_key)
    except:
        return {
            "status": "offline",
            "error": "Stream not active",
            "camera_id": camera_id
        }

    # Generate presigned URL (1 hour expiry)
    playlist_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': playlist_key},
        ExpiresIn=3600
    )

    return {
        "status": "live",
        "camera_id": camera_id,
        "playlist_url": playlist_url
    }


@router.get("/cameras/{camera_id}/alerts")
async def get_alerts(camera_id: str, since: float = 0, limit: int = 50):
    """Get recent alerts with presigned frame URLs."""
    alerts = await db.get_alerts(camera_id, since=since, limit=limit)

    bucket = os.environ['S3_BUCKET']

    # Add presigned URLs for alert frames
    for alert in alerts:
        if alert.get('frame_s3_key'):
            alert['frame_url'] = s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': alert['frame_s3_key']},
                ExpiresIn=3600
            )

    return {"alerts": alerts}
```

### Step 5: Update Dependencies

**requirements.txt**:
```txt
# Web framework
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
mangum>=0.17.0  # Lambda adapter

# Database
asyncpg>=0.29.0

# AWS
boto3>=1.28.0

# Computer Vision
opencv-python-headless>=4.8.0  # Headless for Lambda
ultralytics>=8.0.0  # YOLO
mediapipe>=0.10.0

# LLM
anthropic>=0.7.0

# Utils
python-multipart
pydantic>=2.0.0
pydantic-settings
python-dotenv
bcrypt
python-jose[cryptography]

# Remove:
# aiosqlite
# websockets
```

### Step 6: Create Dockerfile for Lambda

**Dockerfile**:
```dockerfile
FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies
RUN yum install -y \
    libGL \
    libglib2.0-0 \
    && yum clean all

# Copy requirements
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . ${LAMBDA_TASK_ROOT}/

# Lambda handler
CMD ["lambda_segment.handler"]
```

## Testing Checklist

- [ ] PostgreSQL connection works from Lambda
- [ ] RDS Proxy pools connections correctly
- [ ] S3 event triggers Lambda successfully
- [ ] YOLO detection runs on extracted frames
- [ ] Alerts are saved to PostgreSQL
- [ ] Alert frames are uploaded to S3
- [ ] Presigned URLs work for playlist and frames
- [ ] API Gateway routes to Lambda correctly
- [ ] No WebSocket connection attempts
- [ ] Lambda cold start < 5 seconds
- [ ] Concurrent segment processing works

## Migration Commands

```bash
# Install PostgreSQL client
pip install asyncpg

# Run database migration
psql -h <rds-endpoint> -U postgres -d watchtower -f db_migration.sql

# Build Lambda container
docker build -t watchtower-segment-processor .

# Test Lambda locally
docker run -p 9000:8080 watchtower-segment-processor
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d @test-event.json

# Deploy via CDK
cd ../watchtower-infra
cdk deploy ProcessingStack ApiStack
```
