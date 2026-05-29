"""CAPTURE_IMAGE command handler (handled locally on the Pi).

Captures one full frame, encodes JPEG, uploads it to Supabase Storage, and
reports the result back to Express over the WebSocket. No DB writes (Express
persists camera_captures and updates device_commands), no ML, no crop, no
overlay. The whole capture+upload runs under a timeout and never hangs; on any
failure a capture_result FAILED message is sent so Express can mark the command
FAILED.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

import cv2

from . import config, protocol
from .camera_manager import CameraError, CameraManager
from .logging_utils import log
from .storage import StorageClient

WsSender = Callable[[dict[str, Any]], Awaitable[None]]

# Compact, filesystem/URL-safe UTC stamp for the storage object key.
_OBJECT_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def _encode_jpeg(frame) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.CAPTURE_JPEG_QUALITY]
    )
    if not ok:
        raise CameraError("JPEG encoding failed")
    return buffer.tobytes()


def _capture_and_upload(
    camera: CameraManager, storage: StorageClient, command_id: str
) -> dict[str, Any]:
    """Blocking capture -> encode -> upload pipeline. Run in a worker thread.

    Ensures the camera is started (idempotent) so a sensor that was unavailable
    at boot can still recover for a later capture.
    """
    camera.start()
    frame, width, height = camera.capture_full_frame()
    jpeg = _encode_jpeg(frame)

    captured_at = datetime.now(timezone.utc)
    object_path = (
        f"{config.STORAGE_CAPTURE_PREFIX}/"
        f"{captured_at.strftime(_OBJECT_TIMESTAMP_FORMAT)}_{command_id}.jpg"
    )
    storage.upload_jpeg(object_path, jpeg)
    return {
        "image_path": object_path,
        "width": width,
        "height": height,
        "captured_at": captured_at.isoformat(),
    }


async def handle_capture_image(
    command_id: str,
    camera: CameraManager,
    storage: StorageClient,
    ws_send: WsSender,
) -> None:
    """Handle one CAPTURE_IMAGE command end-to-end. Never raises; always reports."""
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_capture_and_upload, camera, storage, command_id),
            timeout=config.CAPTURE_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        error = f"capture timed out after {config.CAPTURE_TIMEOUT_S:g}s"
        log("error", "capture_timeout", command_id=command_id)
        await ws_send(protocol.build_capture_failure(command_id, error))
        return
    except Exception as exc:
        log("error", "capture_failed", command_id=command_id, error=str(exc))
        await ws_send(protocol.build_capture_failure(command_id, str(exc)))
        return

    log(
        "info",
        "capture_succeeded",
        command_id=command_id,
        image_path=result["image_path"],
        width=result["width"],
        height=result["height"],
    )
    await ws_send(
        protocol.build_capture_success(
            command_id,
            result["image_path"],
            result["width"],
            result["height"],
            result["captured_at"],
        )
    )
