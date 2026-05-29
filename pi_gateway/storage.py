"""Supabase Storage client wrapper.

The Pi has Storage access only (no database access). This is the single place
that talks to Supabase, keeping that dependency isolated. Uploads are synchronous
(httpx under the hood), so callers must invoke them from a worker thread to avoid
blocking the asyncio event loop.
"""
from __future__ import annotations

from supabase import Client, create_client

from . import config

JPEG_CONTENT_TYPE = "image/jpeg"


class StorageClient:
    def __init__(self, url: str, key: str, bucket: str) -> None:
        self._bucket = bucket
        self._client: Client = create_client(url, key)

    @classmethod
    def from_config(cls) -> "StorageClient":
        return cls(
            config.SUPABASE_URL,
            config.SUPABASE_STORAGE_KEY,
            config.SUPABASE_STORAGE_BUCKET,
        )

    def upload_jpeg(self, object_path: str, data: bytes) -> str:
        """Upload JPEG bytes to <bucket>/<object_path>; return the object path.

        Raises on failure (network, auth, duplicate object) so the caller can
        report the command FAILED. Blocking; call from a worker thread.
        """
        self._client.storage.from_(self._bucket).upload(
            path=object_path,
            file=data,
            file_options={"content-type": JPEG_CONTENT_TYPE},
        )
        return object_path
