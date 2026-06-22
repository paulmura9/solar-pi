"""Unit tests for CommandDispatcher.dispatch() routing.

These tests verify ONLY the routing decisions of the dispatcher:
  1. CAPTURE_IMAGE is handled locally and is NEVER published to the ESP32 over
     MQTT (the architectural decision that capture stays on the Pi).
  2. ESP32 command types are forwarded to MQTT with the established payload shape.
  3. Unknown command types produce a FAILED command_ack over the WebSocket so a
     command is never left stuck PENDING in the backend.

Nothing real is loaded: the TFLite model, Picamera2 camera, Supabase client and
MQTT client are all replaced with stubs/mocks, so the suite runs on any machine
(no Pi hardware, no cloud credentials).
"""
from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

for _name in (
    "cv2",
    "numpy",
    "picamera2",
    "paho",
    "paho.mqtt",
    "paho.mqtt.client",
    "supabase",
    "ai_edge_litert",
    "ai_edge_litert.interpreter",
    "dotenv",
):
    sys.modules.setdefault(_name, MagicMock(name=_name))

os.environ.setdefault("EXPRESS_WS_URL", "ws://test.invalid/ws/device")
os.environ.setdefault("DEVICE_API_KEY", "test-device-key")
os.environ.setdefault("SUPABASE_URL", "https://test.invalid")
os.environ.setdefault("SUPABASE_STORAGE_KEY", "test-storage-key")
os.environ.setdefault("SUPABASE_STORAGE_BUCKET", "test-bucket")

from pi_gateway import protocol  # noqa: E402
from pi_gateway.camera_manager import CameraError  # noqa: E402
from pi_gateway.dispatcher import CommandDispatcher  # noqa: E402


def _make_dispatcher():
    """Build a dispatcher wired entirely to mocks (vision disabled)."""
    mqtt = MagicMock(name="mqtt")
    camera = MagicMock(name="camera")
    storage = MagicMock(name="storage")
    ws_send = AsyncMock(name="ws_send")
    dispatcher = CommandDispatcher(
        mqtt=mqtt,
        camera=camera,
        storage=storage,
        vision=None,
        ws_send=ws_send,
    )
    return dispatcher, mqtt, camera, storage, ws_send


@pytest.mark.asyncio
async def test_capture_image_is_handled_locally_not_on_mqtt():
    """CAPTURE_IMAGE must be handled on the Pi and never forwarded over MQTT."""
    dispatcher, mqtt, camera, _storage, _ws_send = _make_dispatcher()
    camera.start.side_effect = CameraError("camera unavailable in test")

    await dispatcher.dispatch(
        {
            "id": "cmd-capture-1",
            "payload": {"command_type": protocol.CMD_CAPTURE_IMAGE, "args": {}},
        }
    )

    mqtt.publish_command.assert_not_called()


@pytest.mark.asyncio
async def test_esp32_command_is_forwarded_to_mqtt():
    """An ESP32 command type is published once with the contracted payload shape."""
    dispatcher, mqtt, _camera, _storage, ws_send = _make_dispatcher()
    mqtt.publish_command.return_value = True

    assert protocol.CMD_SET_MODE in protocol.ESP32_COMMAND_TYPES
    args = {"mode": "AUTO"}

    await dispatcher.dispatch(
        {
            "id": "cmd-setmode-1",
            "payload": {"command_type": protocol.CMD_SET_MODE, "args": args},
        }
    )

    mqtt.publish_command.assert_called_once_with(
        {
            "commandId": "cmd-setmode-1",
            "command_type": protocol.CMD_SET_MODE,
            "payload": args,
        }
    )
    ws_send.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_command_type_sends_failed_ack():
    """An unrecognized command type yields a FAILED command_ack and no MQTT publish."""
    dispatcher, mqtt, _camera, _storage, ws_send = _make_dispatcher()

    await dispatcher.dispatch(
        {
            "id": "cmd-bogus-1",
            "payload": {"command_type": "DO_LAUNDRY", "args": {}},
        }
    )

    mqtt.publish_command.assert_not_called()
    ws_send.assert_awaited_once()

    envelope = ws_send.await_args.args[0]
    assert envelope["type"] == protocol.MSG_TYPE_COMMAND_ACK
    payload = envelope["payload"]
    assert payload["commandId"] == "cmd-bogus-1"
    assert payload["status"] == protocol.STATUS_FAILED
    assert "DO_LAUNDRY" in payload["error_message"]
