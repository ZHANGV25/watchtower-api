from __future__ import annotations

import logging

import cv2
import numpy as np

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    mp = None
    HAS_MEDIAPIPE = False
from ultralytics import YOLO

from mask_utils import extract_mask_polygon
from models import BBox, Detection, PoseKeypoint

log = logging.getLogger("watchtower.detector")

# MediaPipe pose landmark names (subset we care about)
_POSE_LANDMARKS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


class Detector:
    """Runs YOLO v8n (detection or segmentation) and MediaPipe Pose on each frame.

    Set WATCHTOWER_SEG=1 to enable segmentation masks. Default is bounding boxes only.
    """

    def __init__(self, yolo_model: str | None = None) -> None:
        if yolo_model is None:
            import os
            yolo_model = "yolov8n-seg.pt" if os.getenv("WATCHTOWER_SEG") == "1" else "yolov8n.pt"
        log.info("Loading YOLO model: %s", yolo_model)
        self._yolo = YOLO(yolo_model)

        self._mp_pose = None
        if HAS_MEDIAPIPE:
            log.info("Initializing MediaPipe Pose")
            self._mp_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0,  # fastest
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            log.info("MediaPipe not available — pose detection disabled")

    def detect(self, frame: np.ndarray, need_pose: bool = False) -> list[Detection]:
        h, w = frame.shape[:2]
        detections: list[Detection] = []

        # YOLO segmentation (imgsz=480 for speed)
        results = self._yolo(frame, verbose=False, imgsz=480)
        result = results[0] if results else None

        yolo_boxes = result.boxes if result else []
        yolo_masks = result.masks if result else None

        # MediaPipe pose only when needed (has pose rules)
        pose_keypoints = None
        if need_pose:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = self._mp_pose.process(rgb)
            pose_keypoints = self._extract_pose(pose_result, w, h)

        for i, box in enumerate(yolo_boxes):
            cls_id = int(box.cls[0])
            cls_name = self._yolo.names.get(cls_id, "unknown")
            conf = float(box.conf[0])

            if conf < 0.4:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            bbox = BBox(
                x=(x1 / w) * 100,
                y=(y1 / h) * 100,
                width=((x2 - x1) / w) * 100,
                height=((y2 - y1) / h) * 100,
            )

            # Extract segmentation polygon
            mask_polygon = extract_mask_polygon(yolo_masks, i, w, h)

            # Attach pose to person detections
            pose = pose_keypoints if cls_name == "person" else None

            detections.append(Detection(
                class_name=cls_name,
                confidence=round(conf, 3),
                bbox=bbox,
                pose=pose,
                mask=mask_polygon,
            ))

        return detections

    def _extract_pose(
        self,
        result: object,
        frame_w: int,
        frame_h: int,
    ) -> list[PoseKeypoint] | None:
        if result is None or not hasattr(result, "pose_landmarks"):
            return None
        if result.pose_landmarks is None:
            return None

        keypoints: list[PoseKeypoint] = []
        for i, lm in enumerate(result.pose_landmarks.landmark):
            name = _POSE_LANDMARKS[i] if i < len(_POSE_LANDMARKS) else f"point_{i}"
            keypoints.append(PoseKeypoint(
                name=name,
                x=lm.x * 100,  # percentage
                y=lm.y * 100,
                visibility=round(lm.visibility, 3),
            ))
        return keypoints
