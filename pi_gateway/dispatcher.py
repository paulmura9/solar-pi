"""Inbound command dispatcher.

Routes commands received from Express (over the WebSocket) by command_type:
  - CAPTURE_IMAGE  -> handled LOCALLY on the Pi (camera capture + Storage upload);
                      never forwarded to the ESP32.
  - ESP32 commands -> forwarded to the ESP32 over MQTT (established firmware
                      contract: topic solar/commands; the ESP32 ACK arrives on
                      solar/commands/ack and is relayed to Express elsewhere).
  - anything else  -> reported FAILED so Express never leaves it stuck.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from . import protocol
from .camera_manager import CameraManager
from .capture import handle_capture_image
from .logging_utils import log
from .mqtt_bridge import MQTTBridge
from .storage import StorageClient
from .vision import DirtDetector

WsSender = Callable[[dict[str, Any]], Awaitable[None]]


class CommandDispatcher:
    def __init__(
        self,
        mqtt: MQTTBridge,
        camera: CameraManager,
        storage: StorageClient,
        vision: Optional[DirtDetector],
        ws_send: WsSender,
    ) -> None:
        self._mqtt = mqtt
        self._camera = camera
        self._storage = storage
        self._vision = vision
        self._ws_send = ws_send

    async def dispatch(self, message: dict[str, Any]) -> None:
        command_id = message.get("id")
        if not isinstance(command_id, str) or not command_id:
            log("warning", "command_missing_id", message=message)
            return

        payload = message.get("payload") or {}
        command_type = payload.get("command_type")
        args = payload.get("args") or {}

        log("info", "command_received", id=command_id, type=command_type)

        if command_type == protocol.CMD_CAPTURE_IMAGE:
            # Handled locally on the Pi; NOT forwarded to the ESP32 over MQTT.
            await handle_capture_image(
                command_id, self._camera, self._storage, self._ws_send, self._vision
            )
            return

        if command_type in protocol.ESP32_COMMAND_TYPES:
            await self._forward_to_esp32(command_id, command_type, args)
            return

        log("warning", "command_unknown_type", id=command_id, type=command_type)
        await self._ws_send(
            protocol.build_command_ack(
                command_id,
                protocol.STATUS_FAILED,
                error_message=f"unsupported command_type: {command_type}",
            )
        )

    async def _forward_to_esp32(
        self, command_id: str, command_type: str, args: dict[str, Any]
    ) -> None:
        # MQTT command/ACK payload shape is the established contract with the
        # ESP32 firmware (preserved from the original gateway). The ESP32 replies
        # on solar/commands/ack, which the gateway relays to Express.
        mqtt_payload = {
            "commandId": command_id,
            "command_type": command_type,
            "payload": args,
        }
        if self._mqtt.publish_command(mqtt_payload):
            return

        await self._ws_send(
            protocol.build_command_ack(
                command_id,
                protocol.STATUS_FAILED,
                error_message="MQTT publish failed on gateway",
            )
        )
