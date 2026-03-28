"""DynamoDB database layer — drop-in replacement for database.py (SQLite).

Use by setting WATCHTOWER_DB_BACKEND=dynamodb. All functions have the
same signatures as database.py so main.py doesn't need to change.
"""
from __future__ import annotations

import json
import logging
import os
import time
from decimal import Decimal

import boto3
from boto3.dynamodb.conditions import Key

from models import Alert, Camera, Condition, Rule, User, Zone, MemoryEntry

log = logging.getLogger("watchtower.database_dynamo")

REGION = os.getenv("AWS_REGION", "us-east-1")
_dynamo = boto3.resource("dynamodb", region_name=REGION)

_cameras = _dynamo.Table("watchtower-cameras")
_alerts = _dynamo.Table("watchtower-alerts")
_rules = _dynamo.Table("watchtower-rules")
_zones = _dynamo.Table("watchtower-zones")
_users = _dynamo.Table("watchtower-users")


def _to_decimal(obj):
    """Convert floats to Decimal for DynamoDB."""
    if isinstance(obj, float):
        return Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _to_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_decimal(i) for i in obj]
    return obj


def _from_decimal(obj):
    """Convert Decimals back to floats."""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _from_decimal(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_decimal(i) for i in obj]
    return obj


# ---------------------------------------------------------------------------
# Init (no-op for DynamoDB — tables already exist)
# ---------------------------------------------------------------------------

async def init_db():
    log.info("Using DynamoDB backend")
    return None

async def get_db():
    return None

async def close_db():
    pass


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------

async def create_camera(cam: Camera) -> Camera:
    _cameras.put_item(Item=_to_decimal(cam.model_dump()))
    return cam

async def get_camera(camera_id: str) -> Camera | None:
    resp = _cameras.get_item(Key={"id": camera_id})
    item = resp.get("Item")
    return Camera(**_from_decimal(item)) if item else None

async def list_cameras() -> list[Camera]:
    resp = _cameras.scan()
    return [Camera(**_from_decimal(i)) for i in resp.get("Items", [])]

async def update_camera(camera_id: str, **kwargs) -> Camera | None:
    expr_parts, values = [], {}
    for i, (k, v) in enumerate(kwargs.items()):
        expr_parts.append(f"#{k} = :v{i}")
        values[f":v{i}"] = _to_decimal(v)
    expr_names = {f"#{k}": k for k in kwargs}
    _cameras.update_item(
        Key={"id": camera_id},
        UpdateExpression="SET " + ", ".join(expr_parts),
        ExpressionAttributeNames=expr_names,
        ExpressionAttributeValues=values,
    )
    return await get_camera(camera_id)

async def delete_camera(camera_id: str) -> bool:
    _cameras.delete_item(Key={"id": camera_id})
    # Cascade: delete zones, rules, alerts
    for zone in await list_zones(camera_id):
        await delete_zone(zone.id)
    await delete_rules_for_camera(camera_id)
    await delete_alerts_for_camera(camera_id)
    return True

async def camera_heartbeat(camera_id: str) -> None:
    _cameras.update_item(
        Key={"id": camera_id},
        UpdateExpression="SET #s = :s, last_seen = :ls",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":s": "online", ":ls": _to_decimal(time.time())},
    )

async def camera_offline(camera_id: str) -> None:
    _cameras.update_item(
        Key={"id": camera_id},
        UpdateExpression="SET #s = :s",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":s": "offline"},
    )


# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------

async def create_zone(zone: Zone) -> Zone:
    _zones.put_item(Item=_to_decimal(zone.model_dump()))
    return zone

async def list_zones(camera_id: str) -> list[Zone]:
    resp = _zones.query(IndexName="camera-index", KeyConditionExpression=Key("camera_id").eq(camera_id))
    return [Zone(**_from_decimal(i)) for i in resp.get("Items", [])]

async def update_zone(zone_id: str, **kwargs) -> None:
    expr_parts, values = [], {}
    for i, (k, v) in enumerate(kwargs.items()):
        expr_parts.append(f"#{k} = :v{i}")
        values[f":v{i}"] = _to_decimal(v)
    expr_names = {f"#{k}": k for k in kwargs}
    _zones.update_item(
        Key={"id": zone_id},
        UpdateExpression="SET " + ", ".join(expr_parts),
        ExpressionAttributeNames=expr_names,
        ExpressionAttributeValues=values,
    )

async def delete_zone(zone_id: str) -> bool:
    _zones.delete_item(Key={"id": zone_id})
    return True

async def replace_zones(camera_id: str, zones: list[Zone]) -> None:
    # Delete existing
    existing = await list_zones(camera_id)
    for z in existing:
        _zones.delete_item(Key={"id": z.id})
    # Insert new
    for z in zones:
        z.camera_id = camera_id
        _zones.put_item(Item=_to_decimal(z.model_dump()))


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

async def create_rule(rule: Rule) -> Rule:
    item = rule.model_dump()
    item["conditions_json"] = json.dumps([c.model_dump() if hasattr(c, "model_dump") else c for c in rule.conditions])
    del item["conditions"]
    _rules.put_item(Item=_to_decimal(item))
    return rule

async def list_rules(camera_id: str) -> list[Rule]:
    resp = _rules.query(IndexName="camera-index", KeyConditionExpression=Key("camera_id").eq(camera_id))
    result = []
    for i in resp.get("Items", []):
        i = _from_decimal(i)
        conditions = [Condition(**c) for c in json.loads(i.pop("conditions_json", "[]"))]
        i["conditions"] = conditions
        i["enabled"] = bool(i.get("enabled", True))
        result.append(Rule(**i))
    return result

async def update_rule(rule_id: str, **kwargs) -> None:
    if "conditions" in kwargs:
        kwargs["conditions_json"] = json.dumps([c.model_dump() if hasattr(c, "model_dump") else c for c in kwargs.pop("conditions")])
    if "enabled" in kwargs:
        kwargs["enabled"] = bool(kwargs["enabled"])
    expr_parts, values, names = [], {}, {}
    for i, (k, v) in enumerate(kwargs.items()):
        expr_parts.append(f"#k{i} = :v{i}")
        values[f":v{i}"] = _to_decimal(v)
        names[f"#k{i}"] = k
    _rules.update_item(Key={"id": rule_id}, UpdateExpression="SET " + ", ".join(expr_parts),
                       ExpressionAttributeNames=names, ExpressionAttributeValues=values)

async def delete_rule(rule_id: str) -> bool:
    _rules.delete_item(Key={"id": rule_id})
    return True

async def delete_rules_for_camera(camera_id: str) -> None:
    rules = await list_rules(camera_id)
    for r in rules:
        _rules.delete_item(Key={"id": r.id})


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

async def create_alert(alert: Alert, frame_path: str = "") -> Alert:
    item = alert.model_dump()
    item["detections_json"] = json.dumps([d.model_dump() for d in alert.detections])
    del item["detections"]
    if frame_path:
        item["frame_path"] = frame_path
    # Remove frame_b64 from DB (too large)
    item.pop("frame_b64", None)
    _alerts.put_item(Item=_to_decimal(item))
    return alert

async def list_alerts(camera_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
    resp = _alerts.query(
        IndexName="camera-index",
        KeyConditionExpression=Key("camera_id").eq(camera_id),
        ScanIndexForward=False,
        Limit=limit + offset,
    )
    items = [_from_decimal(i) for i in resp.get("Items", [])]
    # Sort by timestamp descending
    items.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return items[offset:offset + limit]

async def get_alert(alert_id: str) -> dict | None:
    resp = _alerts.get_item(Key={"id": alert_id})
    item = resp.get("Item")
    return _from_decimal(item) if item else None

async def delete_alerts_for_camera(camera_id: str) -> None:
    alerts = await list_alerts(camera_id, limit=1000)
    for a in alerts:
        _alerts.delete_item(Key={"id": a["id"]})

async def count_alerts(camera_id: str) -> int:
    resp = _alerts.query(
        IndexName="camera-index",
        KeyConditionExpression=Key("camera_id").eq(camera_id),
        Select="COUNT",
    )
    return resp.get("Count", 0)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

async def create_user(user: User) -> User:
    _users.put_item(Item=_to_decimal(user.model_dump()))
    return user

async def get_user_by_username(username: str) -> User | None:
    resp = _users.query(
        IndexName="username-index",
        KeyConditionExpression=Key("username").eq(username),
    )
    items = resp.get("Items", [])
    return User(**_from_decimal(items[0])) if items else None

async def get_user(user_id: str) -> User | None:
    resp = _users.get_item(Key={"id": user_id})
    item = resp.get("Item")
    return User(**_from_decimal(item)) if item else None


# ---------------------------------------------------------------------------
# Memory Entries (using alerts table with special rule_id)
# ---------------------------------------------------------------------------

async def create_memory_entry(camera_id: str, entry: MemoryEntry) -> None:
    # Store as a special alert-like record
    _alerts.put_item(Item=_to_decimal({
        "id": entry.id,
        "camera_id": camera_id,
        "rule_id": "__memory__",
        "rule_name": "Scene Memory",
        "severity": "none",
        "timestamp": entry.timestamp,
        "narration": entry.summary,
        "detections_json": "[]",
        "detection_count": entry.detection_count,
        "created_at": time.time(),
    }))

async def list_memory_entries(camera_id: str, start_time: float = 0, end_time: float = 0, limit: int = 200) -> list[MemoryEntry]:
    if end_time == 0:
        end_time = time.time()
    # Query all alerts for camera, filter for memory entries
    resp = _alerts.query(
        IndexName="camera-index",
        KeyConditionExpression=Key("camera_id").eq(camera_id),
        Limit=500,
    )
    items = [_from_decimal(i) for i in resp.get("Items", []) if i.get("rule_id") == "__memory__"]
    items = [i for i in items if start_time <= i.get("timestamp", 0) <= end_time]
    items.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return [MemoryEntry(
        id=i["id"], camera_id=camera_id, timestamp=i["timestamp"],
        summary=i.get("narration", ""), detection_count=i.get("detection_count", 0),
    ) for i in items[:limit]]
