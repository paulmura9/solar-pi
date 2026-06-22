"""CAPTURE_IMAGE command handler (handled locally on the Pi).

Captures one full frame, encodes JPEG, uploads it to Supabase Storage, and
reports the result back to Express over the WebSocket. No DB writes (Express
persists camera_captures and updates device_commands), no crop, no overlay. The
whole capture+upload runs under a timeout and never hangs; on any failure a
capture_result FAILED message is sent so Express can mark the command FAILED.

When the dirt detector is loaded, the manual capture additionally runs inference
on the same frame and emits a vision_result (best-effort, in addition to the
unchanged camera_capture_result) so a frontend-triggered capture also passes
through the model.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

import cv2

from . import config, protocol
from .camera_manager import CameraError, CameraManager
from .logging_utils import log
from .storage import StorageClient
from .vision import DirtDetector, detect_and_report

WsSender = Callable[[dict[str, Any]], Awaitable[None]]

_OBJECT_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


def _encode_jpeg(frame) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.CAPTURE_JPEG_QUALITY]
    )
    if not ok:
        raise CameraError("JPEG encoding failed")
    return buffer.tobytes()


def capture_and_upload(
    camera: CameraManager,
    storage: StorageClient,
    object_prefix: str,
    name_suffix: str = "",
) -> dict[str, Any]:
    """Blocking capture -> encode -> upload pipeline. Run in a worker thread.

    Shared by the manual CAPTURE_IMAGE handler (prefix "captures", suffix the
    command id) and the periodic vision loop (prefix "vision", no suffix). The
    captured BGR frame is returned so the caller can run inference on it without
    a second capture. Ensures the camera is started (idempotent) so a sensor that
    was unavailable at boot can still recover for a later capture.
    """
    camera.start()
    frame, width, height = camera.capture_full_frame()
    jpeg = _encode_jpeg(frame)

    captured_at = datetime.now(timezone.utc)
    object_path = (
        f"{object_prefix}/"
        f"{captured_at.strftime(_OBJECT_TIMESTAMP_FORMAT)}{name_suffix}.jpg"
    )
    storage.upload_jpeg(object_path, jpeg)
    return {
        "image_path": object_path,
        "width": width,
        "height": height,
        "captured_at": captured_at.isoformat(),
        "frame": frame,
    }


async def send_capture_success(
    command_id: str, result: dict[str, Any], ws_send: WsSender
) -> None:
    """Send a camera_capture_result SUCCESS for an already captured+uploaded frame.

    Shared by the manual CAPTURE_IMAGE handler and the periodic vision loop so the
    builder and message shape are identical. The loop emits this in addition to
    the vision_result so an automatic capture also appears in the frontend's "Last
    Captured Image", which reads camera_capture_result, not vision_results.
    `result` is the dict returned by capture_and_upload.
    """
    await ws_send(
        protocol.build_capture_success(
            command_id,
            result["image_path"],
            result["width"],
            result["height"],
            result["captured_at"],
        )
    )


async def handle_capture_image(
    command_id: str,
    camera: CameraManager,
    storage: StorageClient,
    ws_send: WsSender,
    vision: Optional[DirtDetector],
) -> None:
    """Handle one CAPTURE_IMAGE command end-to-end. Never raises; always reports."""
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                capture_and_upload,
                camera,
                storage,
                config.STORAGE_CAPTURE_PREFIX,
                f"_{command_id}",
            ),
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
    await send_capture_success(command_id, result, ws_send)

    if vision is None:
        return
    try:
        vision_result = await detect_and_report(
            vision, result["frame"], result["image_path"], result["captured_at"], storage, ws_send
        )
    except Exception as exc:
        log("warning", "manual_vision_failed", command_id=command_id, error=str(exc))
        return
    if vision_result is None:
        return
    log(
        "info",
        "manual_vision_done",
        command_id=command_id,
        predicted_class=vision_result.predicted_class,
        dirt_level_percent=vision_result.dirt_level_percent,
        cleaning_required=vision_result.cleaning_required,
    )
