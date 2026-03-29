"""PostgreSQL database adapter using asyncpg for Lambda + RDS."""

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
    """Get or create connection pool.

    For Lambda, this pool is cached across invocations for better performance.
    """
    global _pool

    if _pool is None:
        # Get database credentials from environment or Secrets Manager
        if 'DB_SECRET_ARN' in os.environ:
            # Production: Use Secrets Manager
            secrets_client = boto3.client('secretsmanager')
            secret_arn = os.environ['DB_SECRET_ARN']

            secret_value = secrets_client.get_secret_value(SecretId=secret_arn)
            secret = json.loads(secret_value['SecretString'])

            db_host = os.environ.get('DB_HOST', secret.get('host'))
            db_user = secret.get('username')
            db_password = secret.get('password')
        else:
            # Local development: Use environment variables
            db_host = os.environ.get('DB_HOST', 'localhost')
            db_user = os.environ.get('DB_USER', 'postgres')
            db_password = os.environ.get('DB_PASSWORD', 'postgres')

        db_name = os.environ.get('DB_NAME', 'watchtower')
        db_port = int(os.environ.get('DB_PORT', 5432))

        _pool = await asyncpg.create_pool(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password,
            min_size=1,
            max_size=5,  # Per Lambda container
            command_timeout=60,
        )

        print(f"✓ Database pool created: {db_host}:{db_port}/{db_name}")

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

    print("✓ Database schema initialized")


# ============================================================================
# CRUD Operations
# ============================================================================

# Cameras

async def create_camera(camera: Dict[str, Any]) -> str:
    """Create new camera."""
    pool = await get_pool()
    camera_id = camera.get('id', str(uuid.uuid4()))

    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO cameras (id, name, location, status)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT (id) DO UPDATE SET
                   name = EXCLUDED.name,
                   location = EXCLUDED.location""",
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


async def update_camera(camera_id: str, updates: Dict[str, Any]):
    """Update camera fields."""
    pool = await get_pool()

    set_clauses = []
    values = []
    idx = 1

    for key, value in updates.items():
        set_clauses.append(f"{key} = ${idx}")
        values.append(value)
        idx += 1

    values.append(camera_id)

    async with pool.acquire() as conn:
        await conn.execute(
            f"UPDATE cameras SET {', '.join(set_clauses)} WHERE id = ${idx}",
            *values
        )


# Stream Segments

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


# Rules

async def get_rules(camera_id: str) -> List[Dict[str, Any]]:
    """Get enabled rules for a camera."""
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
        # asyncpg automatically parses JSONB to Python objects
        rules.append(rule)

    return rules


async def create_rule(rule: Dict[str, Any]) -> str:
    """Create new rule."""
    pool = await get_pool()
    rule_id = str(uuid.uuid4())

    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO rules
               (id, camera_id, name, natural_language, conditions, severity, enabled)
               VALUES ($1, $2, $3, $4, $5, $6, $7)""",
            rule_id,
            rule['camera_id'],
            rule['name'],
            rule.get('natural_language', ''),
            json.dumps(rule['conditions']),  # Convert to JSON
            rule.get('severity', 'medium'),
            rule.get('enabled', True)
        )

    return rule_id


# Zones

async def get_zones(camera_id: str) -> List[Dict[str, Any]]:
    """Get zones for a camera."""
    pool = await get_pool()

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM zones WHERE camera_id = $1",
            camera_id
        )

    return [dict(row) for row in rows]


async def create_zone(zone: Dict[str, Any]) -> str:
    """Create new zone."""
    pool = await get_pool()
    zone_id = str(uuid.uuid4())

    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO zones
               (id, camera_id, name, x, y, width, height, color)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
            zone_id,
            zone['camera_id'],
            zone['name'],
            zone['x'],
            zone['y'],
            zone['width'],
            zone['height'],
            zone.get('color', '#00ffff')
        )

    return zone_id


# Alerts

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

    alerts = []
    for row in rows:
        alert = dict(row)
        # Parse JSONB detections
        if isinstance(alert.get('detections'), str):
            alert['detections'] = json.loads(alert['detections'])
        alerts.append(alert)

    return alerts


# Memory Entries

async def create_memory_entry(entry: Dict[str, Any]) -> str:
    """Create memory entry."""
    pool = await get_pool()
    entry_id = str(uuid.uuid4())

    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO memory_entries
               (id, camera_id, timestamp, summary, detection_count)
               VALUES ($1, $2, $3, $4, $5)""",
            entry_id,
            entry['camera_id'],
            entry['timestamp'],
            entry['summary'],
            entry.get('detection_count', 0)
        )

    return entry_id
