"""Lambda handler for WatchTower REST API.

Uses Mangum to adapt FastAPI for Lambda + API Gateway.
Only REST routes — no WebSocket, no camera processing, no YOLO.
Builds a minimal FastAPI app with just the REST routes.
"""
from __future__ import annotations

import os

os.environ["WATCHTOWER_DB_BACKEND"] = "dynamodb"
os.environ["WATCHTOWER_NO_CAMERA"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

# Import only the REST routes — no main.py (which loads YOLO)
from routes.auth_routes import router as auth_router
from routes.cameras import router as cameras_router
from routes.zones import router as zones_router
from routes.rules import router as rules_router
from routes.alerts import router as alerts_router
from routes.activity import router as activity_router
from routes.status import router as status_router
from routes.concerns import router as concerns_router
from routes.reports import router as reports_router
from routes.medications import router as medications_router
from routes.investigate import router as investigate_router
from routes.face import router as face_router
from routes.seed import router as seed_router
from routes.webrtc_proxy import router as webrtc_proxy_router

app = FastAPI(title="WatchTower API (Lambda)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(auth_router)
app.include_router(cameras_router)
app.include_router(zones_router)
app.include_router(rules_router)
app.include_router(alerts_router)
app.include_router(activity_router)
app.include_router(status_router)
app.include_router(concerns_router)
app.include_router(reports_router)
app.include_router(medications_router)
app.include_router(investigate_router)
app.include_router(face_router)
app.include_router(seed_router)
app.include_router(webrtc_proxy_router)


# ---------------------------------------------------------------------------
# Clip processing proxy — invokes the clip-processor Lambda asynchronously
# ---------------------------------------------------------------------------
import json
import boto3

_lambda_client = boto3.client("lambda", region_name=os.getenv("AWS_REGION", "us-east-1"))

@app.post("/api/clips/process")
async def process_clip_proxy(body: dict):
    """Proxy clip processing to the dedicated clip-processor Lambda."""
    try:
        _lambda_client.invoke(
            FunctionName="watchtower-clip-processor",
            InvocationType="Event",  # async — don't wait
            Payload=json.dumps(body),
        )
        return {"status": "queued", "message": "Clip processing started"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/")
async def root():
    return {"service": "watchtower-api", "status": "ok"}

handler = Mangum(app, lifespan="off")
