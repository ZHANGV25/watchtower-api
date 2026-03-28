"""Frame and clip storage abstraction.

Supports local filesystem (default) and S3. Selected via WATCHTOWER_STORAGE env var.
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

log = logging.getLogger("watchtower.storage")

STORAGE_BACKEND = os.getenv("WATCHTOWER_STORAGE", "local")
LOCAL_DATA_DIR = os.getenv("WATCHTOWER_DATA_DIR", "./data")
S3_BUCKET = os.getenv("WATCHTOWER_S3_BUCKET", "watchtower-frames")


class FrameStore(ABC):
    @abstractmethod
    async def save_frame(self, key: str, data: bytes) -> str:
        """Save frame bytes, return the path/key for retrieval."""

    @abstractmethod
    async def get_frame_path(self, key: str) -> str:
        """Return a servable path/URL for a stored frame."""

    @abstractmethod
    async def delete_frame(self, key: str) -> None:
        """Delete a stored frame."""


class LocalFrameStore(FrameStore):
    def __init__(self) -> None:
        self._dir = os.path.join(LOCAL_DATA_DIR, "frames")
        os.makedirs(self._dir, exist_ok=True)

    async def save_frame(self, key: str, data: bytes) -> str:
        path = os.path.join(self._dir, f"{key}.jpg")
        with open(path, "wb") as f:
            f.write(data)
        return f"frames/{key}.jpg"

    async def get_frame_path(self, key: str) -> str:
        return f"/data/frames/{key}.jpg"

    async def delete_frame(self, key: str) -> None:
        path = os.path.join(self._dir, f"{key}.jpg")
        if os.path.exists(path):
            os.remove(path)


class S3FrameStore(FrameStore):
    def __init__(self) -> None:
        try:
            import boto3
            self._client = boto3.client("s3")
        except ImportError:
            raise RuntimeError("boto3 is required for S3 storage: pip install boto3")

    async def save_frame(self, key: str, data: bytes) -> str:
        s3_key = f"frames/{key}.jpg"
        self._client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=data,
            ContentType="image/jpeg",
        )
        return s3_key

    async def get_frame_path(self, key: str) -> str:
        s3_key = f"frames/{key}.jpg"
        url = self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET, "Key": s3_key},
            ExpiresIn=3600,
        )
        return url

    async def delete_frame(self, key: str) -> None:
        s3_key = f"frames/{key}.jpg"
        self._client.delete_object(Bucket=S3_BUCKET, Key=s3_key)


def create_frame_store() -> FrameStore:
    if STORAGE_BACKEND == "s3":
        log.info("Using S3 frame storage (bucket: %s)", S3_BUCKET)
        return S3FrameStore()
    log.info("Using local frame storage (%s)", LOCAL_DATA_DIR)
    return LocalFrameStore()
