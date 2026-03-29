"""Camera routes with HLS streaming support."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import boto3
import os
import db

router = APIRouter()
s3_client = boto3.client('s3')


class StreamUrlResponse(BaseModel):
    status: str
    camera_id: str
    playlist_url: str | None = None
    error: str | None = None


class AlertsResponse(BaseModel):
    alerts: list


@router.get("/cameras/{camera_id}/stream-url", response_model=StreamUrlResponse)
async def get_stream_url(camera_id: str):
    """Get presigned URL for HLS playlist.

    Returns a presigned S3 URL that the frontend can use to stream video via HLS.js.
    The playlist.m3u8 file is polled directly by the HLS player - no API involvement.
    """
    bucket = os.environ.get('S3_BUCKET', os.environ.get('WATCHTOWER_S3_BUCKET', 'watchtower-streams'))
    playlist_key = f'live/{camera_id}/playlist.m3u8'

    # Check if camera exists in database
    camera = await db.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    # Check if stream is active (playlist exists in S3)
    try:
        s3_client.head_object(Bucket=bucket, Key=playlist_key)
    except Exception as e:
        return StreamUrlResponse(
            status="offline",
            camera_id=camera_id,
            error="Stream not active - camera may be offline"
        )

    # Generate presigned URL (1 hour expiry - HLS.js will handle refreshing)
    try:
        playlist_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket, 'Key': playlist_key},
            ExpiresIn=3600
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate presigned URL: {str(e)}")

    return StreamUrlResponse(
        status="live",
        camera_id=camera_id,
        playlist_url=playlist_url
    )


@router.get("/cameras/{camera_id}/alerts", response_model=AlertsResponse)
async def get_camera_alerts(
    camera_id: str,
    since: float = 0,
    limit: int = 50
):
    """Get recent alerts for a camera.

    Args:
        camera_id: Camera identifier
        since: Unix timestamp - only return alerts after this time
        limit: Maximum number of alerts to return

    Returns:
        List of alerts with presigned S3 URLs for alert frames
    """
    # Get alerts from database
    alerts = await db.get_alerts(camera_id, since=since, limit=limit)

    bucket = os.environ.get('S3_BUCKET', os.environ.get('WATCHTOWER_S3_BUCKET', 'watchtower-streams'))

    # Add presigned URLs for alert frames
    for alert in alerts:
        if alert.get('frame_s3_key'):
            try:
                alert['frame_url'] = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket, 'Key': alert['frame_s3_key']},
                    ExpiresIn=3600
                )
            except Exception as e:
                print(f"Failed to generate presigned URL for alert frame: {e}")
                alert['frame_url'] = None

    return AlertsResponse(alerts=alerts)


@router.get("/cameras/{camera_id}/status")
async def get_camera_status(camera_id: str):
    """Get camera connection status.

    Checks if camera is streaming by looking for recent segments in S3.
    """
    camera = await db.get_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    bucket = os.environ.get('S3_BUCKET', os.environ.get('WATCHTOWER_S3_BUCKET', 'watchtower-streams'))
    prefix = f'live/{camera_id}/'

    # Check if there are any recent files in S3
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix,
            MaxKeys=1
        )

        is_online = 'Contents' in response and len(response['Contents']) > 0

        return {
            "camera_id": camera_id,
            "status": "online" if is_online else "offline",
            "name": camera.get('name', 'Unknown'),
            "location": camera.get('location', '')
        }
    except Exception as e:
        print(f"Error checking camera status: {e}")
        return {
            "camera_id": camera_id,
            "status": "unknown",
            "error": str(e)
        }
