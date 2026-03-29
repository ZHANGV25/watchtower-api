"""Proxy WebRTC signaling through API Gateway (HTTPS) to the relay (HTTP).

Solves mixed-content blocking: browser on HTTPS can't call HTTP relay directly.
Browser → API Gateway (HTTPS) → Lambda → Relay (HTTP) → Camera
"""
from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/webrtc", tags=["webrtc"])

RELAY_URL = os.getenv("WATCHTOWER_RELAY_URL", "http://3.238.183.131:8081")


@router.post("/offer/{camera_id}")
async def proxy_offer(camera_id: str, request: Request):
    """Proxy SDP offer to the signaling relay."""
    body = await request.json()
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.post(
                f"{RELAY_URL}/offer/{camera_id}",
                json=body,
            )
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except httpx.TimeoutException:
            raise HTTPException(504, "Camera did not respond in time")
        except Exception as e:
            raise HTTPException(502, f"Relay error: {e}")


@router.get("/ice-config")
async def proxy_ice_config():
    """Proxy ICE config from the relay."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"{RELAY_URL}/ice-config")
            return JSONResponse(content=resp.json())
        except Exception:
            # Fallback STUN-only
            return {"iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
            ]}


@router.get("/cameras")
async def proxy_cameras():
    """List cameras connected to the relay."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"{RELAY_URL}/cameras")
            return JSONResponse(content=resp.json())
        except Exception as e:
            raise HTTPException(502, f"Relay error: {e}")
