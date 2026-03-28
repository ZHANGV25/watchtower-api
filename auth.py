"""JWT authentication and password hashing."""
from __future__ import annotations

import hashlib
import hmac
import json
import base64
import os
import time

SECRET_KEY = os.getenv("WATCHTOWER_SECRET", "watchtower-dev-secret-change-in-prod")
TOKEN_EXPIRY_HOURS = 24


def hash_password(password: str) -> str:
    """Hash a password with a salt using SHA-256. Simple but sufficient for hackathon."""
    salt = os.urandom(16).hex()
    hashed = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash."""
    if ":" not in stored_hash:
        return False
    salt, expected = stored_hash.split(":", 1)
    actual = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    return hmac.compare_digest(actual, expected)


def create_token(user_id: str, username: str) -> str:
    """Create a JWT-like token (HMAC-SHA256 signed JSON)."""
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": time.time() + (TOKEN_EXPIRY_HOURS * 3600),
    }
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    signature = hmac.new(SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{signature}"


def verify_token(token: str) -> dict | None:
    """Verify and decode a token. Returns payload dict or None if invalid."""
    try:
        parts = token.split(".", 1)
        if len(parts) != 2:
            return None
        payload_b64, signature = parts
        expected_sig = hmac.new(SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return None
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None
