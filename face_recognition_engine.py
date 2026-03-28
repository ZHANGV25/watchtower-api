"""Face recognition for identifying the primary resident vs visitors.

Uses face_recognition (dlib) to compare detected faces against a stored
reference encoding of the elderly resident.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("watchtower.face_recognition")

# Try to import face_recognition; gracefully degrade if not available
try:
    import face_recognition as fr
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    log.warning("face_recognition not installed — person identification disabled")


class FaceRecognitionEngine:
    """Manages face encodings for resident identification."""

    def __init__(self, data_dir: str = "./data/faces"):
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        # Cache: camera_id -> list of face encodings (numpy arrays)
        self._encodings: dict[str, list[np.ndarray]] = {}
        self._tolerance = 0.6  # face_recognition default
        self._load_all()

    def _encoding_path(self, camera_id: str) -> Path:
        return self._data_dir / f"{camera_id}_encodings.json"

    def _load_all(self):
        """Load all stored encodings from disk."""
        if not FACE_RECOGNITION_AVAILABLE:
            return
        for path in self._data_dir.glob("*_encodings.json"):
            camera_id = path.stem.replace("_encodings", "")
            try:
                with open(path) as f:
                    data = json.load(f)
                self._encodings[camera_id] = [
                    np.array(enc) for enc in data.get("encodings", [])
                ]
                log.info(
                    "Loaded %d face encodings for camera %s",
                    len(self._encodings[camera_id]),
                    camera_id,
                )
            except Exception as e:
                log.error("Failed to load encodings for %s: %s", camera_id, e)

    def register_face(self, camera_id: str, image_bytes: bytes) -> dict:
        """Register the resident's face from an uploaded photo.

        Args:
            camera_id: Camera/room this resident belongs to
            image_bytes: JPEG/PNG image bytes of the resident's face

        Returns:
            {"status": "ok", "faces_found": N} or {"status": "error", "message": "..."}
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return {"status": "error", "message": "face_recognition library not installed"}

        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"status": "error", "message": "Could not decode image"}

        # Convert BGR to RGB (face_recognition uses RGB)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Find faces and get encodings
        face_locations = fr.face_locations(rgb, model="hog")  # hog is faster than cnn
        if not face_locations:
            return {
                "status": "error",
                "message": "No face detected in the photo. Please upload a clear photo showing the resident's face.",
            }

        encodings = fr.face_encodings(rgb, face_locations)

        # Store encodings (we keep all faces found, but typically just 1)
        # If multiple registrations happen, accumulate encodings for better matching
        existing = self._encodings.get(camera_id, [])
        existing.extend(encodings)
        # Keep at most 10 reference encodings
        self._encodings[camera_id] = existing[-10:]

        # Persist to disk as JSON (numpy arrays -> lists)
        data = {
            "camera_id": camera_id,
            "encodings": [enc.tolist() for enc in self._encodings[camera_id]],
            "count": len(self._encodings[camera_id]),
        }
        with open(self._encoding_path(camera_id), "w") as f:
            json.dump(data, f)

        log.info(
            "Registered %d face(s) for camera %s (total: %d)",
            len(encodings),
            camera_id,
            len(self._encodings[camera_id]),
        )

        return {
            "status": "ok",
            "faces_found": len(encodings),
            "total_references": len(self._encodings[camera_id]),
        }

    def has_reference(self, camera_id: str) -> bool:
        """Check if we have a reference face for this camera."""
        return camera_id in self._encodings and len(self._encodings[camera_id]) > 0

    def identify_people(self, camera_id: str, frame: np.ndarray) -> list[dict]:
        """Identify people in a frame against the stored resident reference.

        Args:
            camera_id: Camera to check references for
            frame: BGR image (OpenCV format)

        Returns:
            List of {"label": "resident"|"visitor", "location": (top,right,bottom,left), "confidence": float}
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return []

        if not self.has_reference(camera_id):
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in frame
        # Use smaller image for speed
        small = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
        face_locations = fr.face_locations(small, model="hog")

        if not face_locations:
            return []

        # Scale locations back up
        face_locations = [
            (int(t * 2), int(r * 2), int(b * 2), int(l * 2))
            for t, r, b, l in face_locations
        ]

        # Get encodings for detected faces
        face_encodings = fr.face_encodings(rgb, face_locations)

        reference_encodings = self._encodings[camera_id]
        results = []

        for encoding, location in zip(face_encodings, face_locations):
            # Compare against all reference encodings
            distances = fr.face_distance(reference_encodings, encoding)
            best_distance = float(min(distances)) if len(distances) > 0 else 1.0
            is_resident = best_distance <= self._tolerance

            confidence = max(0, 1.0 - best_distance)

            results.append({
                "label": "resident" if is_resident else "visitor",
                "location": location,  # (top, right, bottom, left)
                "confidence": round(confidence, 2),
                "distance": round(best_distance, 3),
            })

        return results

    def clear_reference(self, camera_id: str):
        """Remove stored face reference for a camera."""
        self._encodings.pop(camera_id, None)
        path = self._encoding_path(camera_id)
        if path.exists():
            path.unlink()
