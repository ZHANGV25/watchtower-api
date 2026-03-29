"""Face registration REST endpoint for identifying the primary resident."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

import db
from middleware import require_auth

router = APIRouter(prefix="/api/cameras/{camera_id}/face", tags=["face"])

_face_engine = None
_face_available = True


def get_face_engine():
    global _face_engine, _face_available
    if not _face_available:
        return None
    if _face_engine is None:
        try:
            from face_recognition_engine import FaceRecognitionEngine, FACE_RECOGNITION_AVAILABLE
            if not FACE_RECOGNITION_AVAILABLE:
                _face_available = False
                return None
            _face_engine = FaceRecognitionEngine()
        except Exception:
            _face_available = False
            return None
    return _face_engine


@router.post("/register")
async def register_face(
    camera_id: str,
    file: UploadFile = File(...),
    user: dict = Depends(require_auth),
):
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    engine = get_face_engine()
    if engine is None:
        raise HTTPException(501, "Face recognition is not available on this server. It requires the face_recognition library (dlib) which is only supported on the camera device, not Lambda.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(400, "Empty file")
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 10MB)")

    result = engine.register_face(camera_id, contents)
    if result["status"] == "error":
        raise HTTPException(422, result["message"])

    return result


@router.get("/status")
async def face_status(
    camera_id: str,
    user: dict = Depends(require_auth),
):
    cam = await db.get_camera(camera_id)
    if not cam:
        raise HTTPException(404, "Camera not found")

    engine = get_face_engine()
    if engine is None:
        return {"camera_id": camera_id, "has_reference": False, "encoding_count": 0, "available": False}

    return {
        "camera_id": camera_id,
        "has_reference": engine.has_reference(camera_id),
        "encoding_count": len(engine._encodings.get(camera_id, [])),
        "available": True,
    }


@router.delete("/reference")
async def clear_face_reference(
    camera_id: str,
    user: dict = Depends(require_auth),
):
    engine = get_face_engine()
    if engine:
        engine.clear_reference(camera_id)
    return {"status": "cleared"}
