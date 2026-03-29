"""Lambda function to process HLS segments from S3.

Triggered by S3 PUT events when new .ts segments are uploaded.
Extracts frames, runs YOLO detection, evaluates rules, and creates alerts.
"""

import json
import os
import time
import tempfile
import uuid
import cv2
import numpy as np
import boto3
from typing import List
import asyncio

# Import processing modules
from detector import Detector
from rule_engine import RuleEngine
from narrator import Narrator
import database_postgres as db


# Initialize singletons (cached across Lambda invocations for performance)
detector = Detector()
rule_engine = RuleEngine()
narrator = Narrator()
s3_client = boto3.client('s3')

print("✓ Lambda initialized")


def extract_frames_from_segment(
    segment_bytes: bytes,
    sample_rate: float = 2.0
) -> List[np.ndarray]:
    """Extract frames from MPEG-TS segment.

    Args:
        segment_bytes: Raw .ts file bytes
        sample_rate: Extract one frame every N seconds

    Returns:
        List of frames as numpy arrays
    """
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as tmp:
        tmp.write(segment_bytes)
        tmp_path = tmp.name

    try:
        # Open with OpenCV
        cap = cv2.VideoCapture(tmp_path)

        if not cap.isOpened():
            print(f"Failed to open video file: {tmp_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 15
        frame_interval = int(fps * sample_rate)

        if frame_interval == 0:
            frame_interval = 1

        frames = []
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames at specified rate
            if frame_num % frame_interval == 0:
                frames.append(frame.copy())

            frame_num += 1

        cap.release()
        print(f"Extracted {len(frames)} frames from segment")
        return frames

    except Exception as e:
        print(f"Error extracting frames: {e}")
        return []
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass


async def process_segment(event, context):
    """Lambda handler for segment processing.

    Triggered by S3 PUT event when new .ts segment is uploaded.
    """
    print(f"Processing segment event: {json.dumps(event, default=str)}")

    # Parse S3 event
    try:
        record = event['Records'][0]
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']  # e.g., "live/camera-123/segment_001.ts"
    except (KeyError, IndexError) as e:
        print(f"Invalid S3 event structure: {e}")
        return {'statusCode': 400, 'body': 'Invalid event'}

    print(f"Processing: s3://{bucket}/{key}")

    # Extract camera ID from S3 key
    parts = key.split('/')
    if len(parts) < 3 or parts[0] != 'live' or not parts[2].endswith('.ts'):
        print(f"Skipping non-segment file: {key}")
        return {'statusCode': 200, 'body': 'Skipped non-segment file'}

    camera_id = parts[1]
    segment_name = parts[2]

    # Skip playlist files
    if segment_name.endswith('.m3u8'):
        return {'statusCode': 200, 'body': 'Skipped playlist'}

    # Download segment from S3
    try:
        print(f"Downloading segment from S3...")
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        segment_bytes = obj['Body'].read()
        print(f"Downloaded {len(segment_bytes)} bytes")
    except Exception as e:
        print(f"Failed to download segment: {e}")
        return {'statusCode': 500, 'body': f'Download failed: {str(e)}'}

    # Extract frames (sample every 2 seconds)
    frames = extract_frames_from_segment(segment_bytes, sample_rate=2.0)

    if not frames:
        print("No frames extracted, skipping")
        return {'statusCode': 200, 'body': 'No frames'}

    # Get rules and zones for this camera
    try:
        rules = await db.get_rules(camera_id)
        zones = await db.get_zones(camera_id)
        print(f"Loaded {len(rules)} rules and {len(zones)} zones")
    except Exception as e:
        print(f"Failed to load rules/zones: {e}")
        rules = []
        zones = []

    # Process each frame
    alert_count = 0
    current_time = time.time()

    for frame_idx, frame in enumerate(frames):
        try:
            # Run YOLO detection
            detections = detector.detect(frame)
            print(f"Frame {frame_idx}: {len(detections)} detections")

            if not rules:
                continue

            # Evaluate rules
            triggered_rules = rule_engine.evaluate(
                camera_id=camera_id,
                detections=detections,
                frame=frame,
                rules=rules,
                zones=zones
            )

            # Create alerts for triggered rules
            for rule in triggered_rules:
                alert_count += 1

                # Upload alert frame to S3
                alert_id = f"{camera_id}_{int(current_time)}_{frame_idx}_{uuid.uuid4().hex[:8]}"
                frame_key = f"alerts/{alert_id}.jpg"

                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                s3_client.put_object(
                    Bucket=bucket,
                    Key=frame_key,
                    Body=buffer.tobytes(),
                    ContentType='image/jpeg'
                )
                print(f"Uploaded alert frame: {frame_key}")

                # Create alert in database
                alert_db_id = await db.create_alert({
                    'camera_id': camera_id,
                    'rule_id': rule.get('id'),
                    'rule_name': rule['name'],
                    'severity': rule['severity'],
                    'timestamp': current_time,
                    'frame_s3_key': frame_key,
                    'detections': detections
                })
                print(f"Created alert: {alert_db_id}")

                # Generate narration asynchronously (non-blocking)
                try:
                    narration = await narrator.narrate(frame, detections, rule)
                    await db.update_alert_narration(alert_db_id, narration)
                    print(f"Added narration to alert")
                except Exception as e:
                    print(f"Narration failed: {e}")

        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            continue

    # Save segment metadata to database
    try:
        await db.create_stream_segment({
            'camera_id': camera_id,
            's3_key': key,
            'timestamp': current_time,
            'processed': True,
            'alert_count': alert_count
        })
        print(f"Saved segment metadata")
    except Exception as e:
        print(f"Failed to save segment metadata: {e}")

    print(f"✓ Processed segment with {alert_count} alerts")

    return {
        'statusCode': 200,
        'body': json.dumps({
            'processed': True,
            'frames': len(frames),
            'alerts': alert_count,
            'camera_id': camera_id
        })
    }


def handler(event, context):
    """Synchronous Lambda handler (wraps async function)."""
    return asyncio.run(process_segment(event, context))
