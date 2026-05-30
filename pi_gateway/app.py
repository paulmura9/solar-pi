"""SolarGateway orchestrator and process entrypoint.

Wires the components together (config -> camera + storage + MQTT + WS ->
dispatcher), owns startup/shutdown, and relays MQTT inbound messages (telemetry,
ESP32 ACKs, events) up to Express. Business logic lives in the dedicated modules;
this file is glue and lifecycle only.
"""
from __future__ import annotations

import asyncio
import signal
from typing import Any

from . import config, protocol
from .camera_manager import CameraError, CameraManager
from .dispatcher import CommandDispatcher
from .logging_utils import log
from .mqtt_bridge import MQTTBridge
from .offline_buffer import OfflineBuffer
from .storage import StorageClient
from .telemetry import validate_telemetry
from .ws_client import WebSocketClient


class SolarGateway:
    def __init__(self) -> None:
        self._buffer = OfflineBuffer(
            db_path=config.BUFFER_DB_PATH,
            max_age_seconds=config.BUFFER_MAX_AGE_HOURS * 3600,
        )
        self._ws = WebSocketClient(
            url=config.EXPRESS_WS_URL,
            api_key=config.DEVICE_API_KEY,
            device_id=config.DEVICE_ID,
            buffer=self._buffer,
            on_command=self._on_command_from_cloud,
        )
        self._mqtt = MQTTBridge(
            host=config.MQTT_BROKER_HOST,
            port=config.MQTT_BROKER_PORT,
            username=config.MQTT_USERNAME,
            password=config.MQTT_PASSWORD,
            on_telemetry=self._forward_telemetry,
            on_ack=self._forward_ack,
            on_event=self._forward_event,
        )
        self._camera = CameraManager()
        self._storage = StorageClient.from_config()
        self._dispatcher = CommandDispatcher(
            mqtt=self._mqtt,
            camera=self._camera,
            storage=self._storage,
            ws_send=self._ws.send_or_buffer,
        )
        self._stop_event = asyncio.Event()

    def stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        log("info", "gateway_starting", device_id=config.DEVICE_ID, ws_url=config.EXPRESS_WS_URL)
        self._start_camera_best_effort()

        loop = asyncio.get_running_loop()
        self._mqtt.start(loop)

        ws_task = asyncio.create_task(self._ws.run())
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        cleanup_task = asyncio.create_task(self._cleanup_loop())

        await self._stop_event.wait()

        log("info", "gateway_stopping")

        # Signal then cancel: the event lets the WS loop exit cleanly if it is
        # between iterations; cancel breaks it out of any reconnect sleep.
        self._ws.stop()
        ws_task.cancel()
        heartbeat_task.cancel()
        cleanup_task.cancel()

        try:
            await asyncio.wait_for(ws_task, timeout=config.WS_SHUTDOWN_TIMEOUT_S)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        self._mqtt.stop()
        self._camera.stop()
        log("info", "gateway_stopped")

    def _start_camera_best_effort(self) -> None:
        """Start the camera but tolerate failure.

        A missing/faulty camera must not take down the ESP32/telemetry path.
        CAPTURE_IMAGE commands will fail cleanly (and retry start) until the
        camera recovers.
        """
        try:
            self._camera.start()
        except CameraError as exc:
            log("error", "camera_unavailable_at_startup", error=str(exc))

    async def _on_command_from_cloud(self, message: dict[str, Any]) -> None:
        await self._dispatcher.dispatch(message)

    async def _forward_telemetry(self, data: dict[str, Any]) -> None:
        valid, reason = validate_telemetry(data)
        if not valid:
            log("warning", "telemetry_rejected", reason=reason)
            return

        await self._ws.send_or_buffer(
            protocol.build_envelope(protocol.MSG_TYPE_TELEMETRY, data)
        )

    async def _forward_ack(self, data: dict[str, Any]) -> None:
        """Relay an ESP32 ACK to Express as a sanitized command_ack envelope."""
        ack = protocol.build_command_ack_from_esp32(data)
        if ack is None:
            log("warning", "ack_invalid", data=data)
            return

        await self._ws.send_or_buffer(ack)
        ack_payload = ack["payload"]
        log(
            "info",
            "command_acked",
            command_id=ack_payload["commandId"],
            status=ack_payload["status"],
            has_ack_payload="ack_payload" in ack_payload,
        )

    async def _forward_event(self, data: dict[str, Any]) -> None:
        await self._ws.send_or_buffer(
            protocol.build_envelope(protocol.MSG_TYPE_ESP32_EVENT, data)
        )

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(config.WS_HEARTBEAT_INTERVAL_S)
                if self._ws.is_connected:
                    await self._ws.send(
                        protocol.build_envelope(
                            protocol.MSG_TYPE_HEARTBEAT,
                            {"camera_ok": self._camera.is_ok()},
                        )
                    )
        except asyncio.CancelledError:
            pass

    async def _cleanup_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(config.BUFFER_CLEANUP_INTERVAL_S)
                removed = await self._buffer.cleanup()
                if removed > 0:
                    log("info", "buffer_cleaned", removed=removed)
        except asyncio.CancelledError:
            pass


async def main() -> None:
    gateway = SolarGateway()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, gateway.stop)

    await gateway.run()
