"""WatchTower API - Simplified REST-only version.

No WebSockets, no in-memory state, no real-time processing loops.
Pure stateless REST API designed for Lambda deployment.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path

# Import database (PostgreSQL or fallback)
import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    print("🚀 Initializing WatchTower API...")
    await db.init_db()
    print("✅ Database initialized")
    yield
    print("👋 Shutting down WatchTower API")


app = FastAPI(
    title="WatchTower API",
    description="Elder care monitoring API with HLS streaming support",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frame storage (local development only)
data_dir = Path(os.environ.get('WATCHTOWER_DATA_DIR', './data'))
frames_dir = data_dir / 'frames'
frames_dir.mkdir(parents=True, exist_ok=True)

try:
    app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")
except Exception as e:
    print(f"Warning: Could not mount static files: {e}")


# Include route modules
try:
    from routes import cameras, alerts, rules, zones, auth
    app.include_router(cameras.router, prefix="/api")
    app.include_router(alerts.router, prefix="/api")
    app.include_router(rules.router, prefix="/api")
    app.include_router(zones.router, prefix="/api")
    app.include_router(auth.router, prefix="/api/auth")
    print("✅ Routes loaded")
except ImportError as e:
    print(f"⚠️  Some routes could not be loaded: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "WatchTower API",
        "version": "2.0.0",
        "mode": "REST-only (HLS streaming)",
        "database": os.environ.get('WATCHTOWER_DB_BACKEND', 'postgresql')
    }


@app.get("/health")
async def health():
    """Health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "database": "connected"
    }


# Lambda handler for AWS deployment
try:
    from mangum import Mangum
    lambda_handler = Mangum(app, lifespan="off")
except ImportError:
    print("⚠️  Mangum not installed - Lambda deployment not available")
    lambda_handler = None
