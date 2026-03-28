"""Face registration REST endpoint for identifying the primary resident."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

import db
from middleware import require_auth

router = APIRouter(prefix="/api/cameras/{camera_id}/face", tags=["face"])

# Lazy singleton for face engine
_face_engine = None


def get_face_engine():
    global _face_engine
    if _face_engine is None:
        from face_recognition_engine import FaceRecognitionEngine
        _face_engine = FaceRecognitionEngine()
    return _face_engine


@router.post("/register")
async def register_face(
    camera_id: str,
    file: UploadFile = File(...),
    user: dict = Depends(require_auth),
):
    """Upload a reference photo of the resident for face recognition.

    The photo should clearly show the resident's face. Multiple photos
    can be uploaded to improve recognition accuracy.
    """
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    # Read image bytes
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(400, "Empty file")
    if len(contents) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(400, "File too large (max 10MB)")

    engine = get_face_engine()
    result = engine.register_face(camera_id, contents)

    if result["status"] == "error":
        raise HTTPException(422, result["message"])

    return result


@router.get("/status")
async def face_status(
    camera_id: str,
    user: dict = Depends(require_auth),
):
    """Check if face recognition is set up for this room."""
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    engine = get_face_engine()
    has_ref = engine.has_reference(camera_id)

    return {
        "camera_id": camera_id,
        "has_reference": has_ref,
        "encoding_count": len(engine._encodings.get(camera_id, [])),
    }


@router.delete("/reference")
async def clear_face_reference(
    camera_id: str,
    user: dict = Depends(require_auth),
):
    """Clear the stored face reference for this room."""
    engine = get_face_engine()
    engine.clear_reference(camera_id)
    return {"status": "cleared"}
