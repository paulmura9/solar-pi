"""Wire protocol between the Pi gateway and the Express backend.

Single source of truth for the JSON message envelopes exchanged over the
authenticated /ws/device WebSocket, plus the command-type vocabulary.

Inbound (Express -> Pi):
    {"type": "command", "id": <command_id>,
     "payload": {"command_type": <UPPERCASE>, "args": {...}}}

Outbound (Pi -> Express) base envelope:
    {"v": 1, "type": <type>, "id": <uuid>, "timestamp": <iso8601>, "payload": {...}}

Outbound message types:
  - "telemetry"       payload = sensor reading forwarded from the ESP32
  - "esp32_event"     payload = raw ESP32 event
  - "heartbeat"       payload = {}
  - "sync_request"    payload = {"last_command_id": <id|null>}
  - "command_ack"     payload = {commandId, status, error_message?, ack_payload?}
                      ESP32-bound command completion and local dispatch failures.
  - "camera_capture_result"  payload = {command_id, status, image_path, width,
                      height, captured_at} on success (status "SUCCESS"), or
                      {command_id, status, error_message} on failure
                      (status "FAILED").

CONTRACT NOTE (camera_capture_result): this message is defined here because the
Express side of the contract is not present in this repo. Express's /ws/device
handler routes on the exact type "camera_capture_result" and strictly validates
the payload: on "SUCCESS", INSERT a camera_captures row (command_id, image_path,
width, height, captured_at) and set device_commands -> ACKNOWLEDGED; on "FAILED",
set device_commands -> FAILED with error_message. The Pi never writes these
tables itself.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

PROTOCOL_VERSION = 1

# Inbound message types (Express -> Pi).
MSG_TYPE_COMMAND = "command"
MSG_TYPE_HEARTBEAT_ACK = "heartbeat_ack"
MSG_TYPE_SERVER_SHUTDOWN = "server_shutting_down"

# Outbound message types (Pi -> Express).
MSG_TYPE_SYNC_REQUEST = "sync_request"
MSG_TYPE_TELEMETRY = "telemetry"
MSG_TYPE_ESP32_EVENT = "esp32_event"
MSG_TYPE_HEARTBEAT = "heartbeat"
MSG_TYPE_COMMAND_ACK = "command_ack"
# Express routes the capture result only on this exact discriminator string.
MSG_TYPE_CAPTURE_RESULT = "camera_capture_result"

# device_commands.status values (the DB write is performed by Express).
# STATUS_ACKNOWLEDGED is the ESP32-ACK / command_ack success value; the capture
# result uses STATUS_CAPTURE_SUCCESS, which Express's success branch expects.
STATUS_ACKNOWLEDGED = "ACKNOWLEDGED"
STATUS_CAPTURE_SUCCESS = "SUCCESS"
STATUS_FAILED = "FAILED"

# Keys consumed directly from a raw ESP32 ACK; everything else is forwarded to
# Express under ack_payload.
_ACK_RESERVED_KEYS = frozenset({"commandId", "status", "message"})

# device_commands.command_type vocabulary (UPPERCASE; see CLAUDE.md).
CMD_SET_MODE = "SET_MODE"
CMD_MOVE_PANEL = "MOVE_PANEL"
CMD_MOVE = "MOVE"
CMD_RESET_POSITION = "RESET_POSITION"
CMD_REQUEST_STATUS = "REQUEST_STATUS"
CMD_START_TRACKING = "START_TRACKING"
CMD_STOP_TRACKING = "STOP_TRACKING"
CMD_TRIGGER_CLEANING = "TRIGGER_CLEANING"
CMD_CAPTURE_IMAGE = "CAPTURE_IMAGE"

# Commands forwarded to the ESP32 over MQTT. CAPTURE_IMAGE is intentionally
# excluded: it is handled locally on the Pi.
ESP32_COMMAND_TYPES = frozenset(
    {
        CMD_SET_MODE,
        CMD_MOVE_PANEL,
        CMD_MOVE,
        CMD_RESET_POSITION,
        CMD_REQUEST_STATUS,
        CMD_START_TRACKING,
        CMD_STOP_TRACKING,
        CMD_TRIGGER_CLEANING,
    }
)


def now_iso() -> str:
    """Current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def build_envelope(message_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a payload in the standard outbound envelope with a unique id."""
    return {
        "v": PROTOCOL_VERSION,
        "type": message_type,
        "id": str(uuid.uuid4()),
        "timestamp": now_iso(),
        "payload": payload,
    }


def build_command_ack(
    command_id: str,
    status: str,
    error_message: Optional[str] = None,
    ack_payload: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a command_ack envelope for ESP32-bound commands / local failures."""
    payload: dict[str, Any] = {"commandId": command_id, "status": status}
    if error_message is not None:
        payload["error_message"] = error_message
    if ack_payload:
        payload["ack_payload"] = ack_payload
    return build_envelope(MSG_TYPE_COMMAND_ACK, payload)


def build_capture_success(
    command_id: str,
    image_path: str,
    width: int,
    height: int,
    captured_at: str,
) -> dict[str, Any]:
    """Build a successful capture_result envelope (no DB write happens on the Pi)."""
    return build_envelope(
        MSG_TYPE_CAPTURE_RESULT,
        {
            "command_id": command_id,
            "status": STATUS_CAPTURE_SUCCESS,
            "image_path": image_path,
            "width": width,
            "height": height,
            "captured_at": captured_at,
        },
    )


def build_capture_failure(command_id: str, error_message: str) -> dict[str, Any]:
    """Build a failed capture_result envelope so Express can mark the command FAILED."""
    return build_envelope(
        MSG_TYPE_CAPTURE_RESULT,
        {
            "command_id": command_id,
            "status": STATUS_FAILED,
            "error_message": error_message,
        },
    )


def build_command_ack_from_esp32(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Translate a raw ESP32 ACK into a command_ack envelope.

    Pure (no I/O, no logging). Returns None when the ACK is malformed - a
    missing/blank commandId or a status that is not ACKNOWLEDGED/FAILED - so the
    caller can drop it.

    Wire contract from the ESP32:
      - commandId (required, string)
      - status (required, ACKNOWLEDGED or FAILED)
      - message (optional, only on FAILED) -> forwarded as error_message
      - any other keys -> forwarded under ack_payload for the backend.
    """
    command_id = data.get("commandId")
    if not isinstance(command_id, str) or not command_id:
        return None

    status = data.get("status")
    if status not in (STATUS_ACKNOWLEDGED, STATUS_FAILED):
        return None

    message_field = data.get("message")
    error_message = (
        message_field
        if status == STATUS_FAILED and isinstance(message_field, str)
        else None
    )
    extra = {k: v for k, v in data.items() if k not in _ACK_RESERVED_KEYS}
    return build_command_ack(
        command_id,
        status,
        error_message=error_message,
        ack_payload=extra or None,
    )
