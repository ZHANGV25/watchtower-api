"""SQLite database layer with async CRUD operations.

All WatchTower state (cameras, zones, rules, alerts, users, memory)
persists here. In-memory state in main.py is a cache loaded from this
DB at startup and written through on every mutation.
"""
from __future__ import annotations

import json
import logging
import os
import time

import aiosqlite

from models import Alert, Camera, Condition, Detection, Rule, User, Zone, MemoryEntry

log = logging.getLogger("watchtower.database")

DB_PATH = os.getenv("WATCHTOWER_DB", "./data/watchtower.db")

_db: aiosqlite.Connection | None = None


async def init_db() -> aiosqlite.Connection:
    """Initialize database, create tables, return connection."""
    global _db
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    _db = await aiosqlite.connect(DB_PATH)
    _db.row_factory = aiosqlite.Row
    await _db.execute("PRAGMA journal_mode=WAL")
    await _db.execute("PRAGMA foreign_keys=ON")

    await _db.executescript("""
        CREATE TABLE IF NOT EXISTS cameras (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT DEFAULT '',
            location TEXT DEFAULT '',
            status TEXT DEFAULT 'offline',
            last_seen REAL DEFAULT 0,
            webrtc_url TEXT DEFAULT '',
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS zones (
            id TEXT PRIMARY KEY,
            camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            width REAL NOT NULL,
            height REAL NOT NULL,
            color TEXT DEFAULT '#22d3ee',
            created_at REAL NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS rules (
            id TEXT PRIMARY KEY,
            camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            natural_language TEXT NOT NULL,
            conditions_json TEXT NOT NULL DEFAULT '[]',
            severity TEXT DEFAULT 'medium',
            enabled INTEGER DEFAULT 1,
            created_at REAL NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
            rule_id TEXT NOT NULL,
            rule_name TEXT NOT NULL,
            severity TEXT NOT NULL,
            timestamp REAL NOT NULL,
            frame_path TEXT DEFAULT '',
            narration TEXT DEFAULT '',
            detections_json TEXT DEFAULT '[]',
            created_at REAL NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memory_entries (
            id TEXT PRIMARY KEY,
            camera_id TEXT NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
            timestamp REAL NOT NULL,
            summary TEXT NOT NULL,
            detection_count INTEGER DEFAULT 0,
            frame_url TEXT DEFAULT '',
            created_at REAL NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_zones_camera ON zones(camera_id);
        CREATE INDEX IF NOT EXISTS idx_rules_camera ON rules(camera_id);
        CREATE INDEX IF NOT EXISTS idx_alerts_camera ON alerts(camera_id);
        CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp);
        CREATE INDEX IF NOT EXISTS idx_memory_camera ON memory_entries(camera_id);
        CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory_entries(timestamp);
    """)
    await _db.commit()
    log.info("Database initialized at %s", DB_PATH)
    return _db


async def get_db() -> aiosqlite.Connection:
    if _db is None:
        return await init_db()
    return _db


async def close_db() -> None:
    global _db
    if _db:
        await _db.close()
        _db = None


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

async def create_camera(cam: Camera) -> Camera:
    db = await get_db()
    await db.execute(
        "INSERT OR IGNORE INTO cameras (id, name, description, location, status, last_seen, webrtc_url, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (cam.id, cam.name, cam.description, cam.location, cam.status, cam.last_seen, cam.webrtc_url, cam.created_at),
    )
    await db.commit()
    return cam


async def get_camera(camera_id: str) -> Camera | None:
    db = await get_db()
    async with db.execute("SELECT * FROM cameras WHERE id = ?", (camera_id,)) as cur:
        row = await cur.fetchone()
        if not row:
            return None
        return Camera(**dict(row))


async def list_cameras() -> list[Camera]:
    db = await get_db()
    async with db.execute("SELECT * FROM cameras ORDER BY created_at DESC") as cur:
        rows = await cur.fetchall()
        return [Camera(**dict(r)) for r in rows]


async def update_camera(camera_id: str, **kwargs) -> Camera | None:
    db = await get_db()
    fields = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [camera_id]
    await db.execute(f"UPDATE cameras SET {fields} WHERE id = ?", values)
    await db.commit()
    return await get_camera(camera_id)


async def delete_camera(camera_id: str) -> bool:
    db = await get_db()
    cur = await db.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
    await db.commit()
    return cur.rowcount > 0


async def camera_heartbeat(camera_id: str) -> None:
    db = await get_db()
    await db.execute(
        "UPDATE cameras SET status = 'online', last_seen = ? WHERE id = ?",
        (time.time(), camera_id),
    )
    await db.commit()


async def camera_offline(camera_id: str) -> None:
    db = await get_db()
    await db.execute("UPDATE cameras SET status = 'offline' WHERE id = ?", (camera_id,))
    await db.commit()


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------

async def create_zone(zone: Zone) -> Zone:
    db = await get_db()
    await db.execute(
        "INSERT OR REPLACE INTO zones (id, camera_id, name, x, y, width, height, color, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (zone.id, zone.camera_id, zone.name, zone.x, zone.y, zone.width, zone.height, zone.color, time.time()),
    )
    await db.commit()
    return zone


async def list_zones(camera_id: str) -> list[Zone]:
    db = await get_db()
    async with db.execute("SELECT * FROM zones WHERE camera_id = ?", (camera_id,)) as cur:
        rows = await cur.fetchall()
        return [Zone(**dict(r)) for r in rows]


async def update_zone(zone_id: str, **kwargs) -> None:
    db = await get_db()
    fields = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [zone_id]
    await db.execute(f"UPDATE zones SET {fields} WHERE id = ?", values)
    await db.commit()


async def delete_zone(zone_id: str) -> bool:
    db = await get_db()
    cur = await db.execute("DELETE FROM zones WHERE id = ?", (zone_id,))
    await db.commit()
    return cur.rowcount > 0


async def replace_zones(camera_id: str, zones: list[Zone]) -> None:
    """Replace all zones for a camera (used by auto-detect and bootstrap)."""
    db = await get_db()
    await db.execute("DELETE FROM zones WHERE camera_id = ?", (camera_id,))
    for z in zones:
        z.camera_id = camera_id
        await db.execute(
            "INSERT INTO zones (id, camera_id, name, x, y, width, height, color, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (z.id, z.camera_id, z.name, z.x, z.y, z.width, z.height, z.color, time.time()),
        )
    await db.commit()


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

async def create_rule(rule: Rule) -> Rule:
    db = await get_db()
    conditions_json = json.dumps([c.model_dump() for c in rule.conditions])
    await db.execute(
        "INSERT OR REPLACE INTO rules (id, camera_id, name, natural_language, conditions_json, severity, enabled, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (rule.id, rule.camera_id, rule.name, rule.natural_language, conditions_json, rule.severity, int(rule.enabled), rule.created_at),
    )
    await db.commit()
    return rule


async def list_rules(camera_id: str) -> list[Rule]:
    db = await get_db()
    async with db.execute("SELECT * FROM rules WHERE camera_id = ?", (camera_id,)) as cur:
        rows = await cur.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["conditions"] = [Condition(**c) for c in json.loads(d.pop("conditions_json"))]
            d["enabled"] = bool(d["enabled"])
            result.append(Rule(**d))
        return result


async def update_rule(rule_id: str, **kwargs) -> None:
    db = await get_db()
    if "conditions" in kwargs:
        kwargs["conditions_json"] = json.dumps([c.model_dump() if hasattr(c, "model_dump") else c for c in kwargs.pop("conditions")])
    if "enabled" in kwargs:
        kwargs["enabled"] = int(kwargs["enabled"])
    fields = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [rule_id]
    await db.execute(f"UPDATE rules SET {fields} WHERE id = ?", values)
    await db.commit()


async def delete_rule(rule_id: str) -> bool:
    db = await get_db()
    cur = await db.execute("DELETE FROM rules WHERE id = ?", (rule_id,))
    await db.commit()
    return cur.rowcount > 0


async def delete_rules_for_camera(camera_id: str) -> None:
    db = await get_db()
    await db.execute("DELETE FROM rules WHERE camera_id = ?", (camera_id,))
    await db.commit()


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

async def create_alert(alert: Alert, frame_path: str = "") -> Alert:
    db = await get_db()
    detections_json = json.dumps([d.model_dump() for d in alert.detections])
    await db.execute(
        "INSERT INTO alerts (id, camera_id, rule_id, rule_name, severity, timestamp, frame_path, narration, detections_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (alert.id, alert.camera_id, alert.rule_id, alert.rule_name, alert.severity, alert.timestamp, frame_path, alert.narration, detections_json, time.time()),
    )
    await db.commit()
    return alert


async def list_alerts(camera_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
    db = await get_db()
    async with db.execute(
        "SELECT * FROM alerts WHERE camera_id = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        (camera_id, limit, offset),
    ) as cur:
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


async def get_alert(alert_id: str) -> dict | None:
    db = await get_db()
    async with db.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)) as cur:
        row = await cur.fetchone()
        return dict(row) if row else None


async def delete_alerts_for_camera(camera_id: str) -> None:
    db = await get_db()
    await db.execute("DELETE FROM alerts WHERE camera_id = ?", (camera_id,))
    await db.commit()


async def count_alerts(camera_id: str) -> int:
    db = await get_db()
    async with db.execute("SELECT COUNT(*) FROM alerts WHERE camera_id = ?", (camera_id,)) as cur:
        row = await cur.fetchone()
        return row[0] if row else 0


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

async def create_user(user: User) -> User:
    db = await get_db()
    await db.execute(
        "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
        (user.id, user.username, user.password_hash, user.created_at),
    )
    await db.commit()
    return user


async def get_user_by_username(username: str) -> User | None:
    db = await get_db()
    async with db.execute("SELECT * FROM users WHERE username = ?", (username,)) as cur:
        row = await cur.fetchone()
        if not row:
            return None
        return User(**dict(row))


async def get_user(user_id: str) -> User | None:
    db = await get_db()
    async with db.execute("SELECT * FROM users WHERE id = ?", (user_id,)) as cur:
        row = await cur.fetchone()
        if not row:
            return None
        return User(**dict(row))


# ---------------------------------------------------------------------------
# Memory Entries
# ---------------------------------------------------------------------------

async def create_memory_entry(camera_id: str, entry: MemoryEntry) -> None:
    db = await get_db()
    await db.execute(
        "INSERT INTO memory_entries (id, camera_id, timestamp, summary, detection_count, frame_url, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (entry.id, camera_id, entry.timestamp, entry.summary, entry.detection_count, entry.frame_url or "", time.time()),
    )
    await db.commit()


async def list_memory_entries(camera_id: str, start_time: float = 0, end_time: float = 0, limit: int = 200) -> list[MemoryEntry]:
    db = await get_db()
    if end_time == 0:
        end_time = time.time()
    async with db.execute(
        "SELECT * FROM memory_entries WHERE camera_id = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT ?",
        (camera_id, start_time, end_time, limit),
    ) as cur:
        rows = await cur.fetchall()
        return [MemoryEntry(**dict(r)) for r in rows]
