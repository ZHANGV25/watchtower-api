"""FastAPI authentication dependencies."""
from __future__ import annotations

from fastapi import Header, HTTPException, Query

from auth import verify_token


async def require_auth(authorization: str = Header(None)) -> dict:
    """Dependency that validates JWT from Authorization header.

    Usage: @router.get("/...", dependencies=[Depends(require_auth)])
    Or: async def handler(user: dict = Depends(require_auth))
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload


def verify_ws_token(token: str | None) -> dict | None:
    """Verify a token passed as WebSocket query parameter.

    Returns payload dict or None. Does not raise — WebSocket should
    close gracefully on auth failure.
    """
    if not token:
        return None
    return verify_token(token)
