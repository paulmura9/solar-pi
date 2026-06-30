"""MQTT bridge to the local ESP32 (Mosquitto broker, never exposed publicly).

Subscribes to telemetry / ACK / event topics and dispatches them to injected
async handlers, and publishes ESP32-bound commands. paho-mqtt runs its own
network thread, so inbound messages are handed to the asyncio loop via
run_coroutine_threadsafe, with a backpressure counter to bound memory.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Awaitable, Callable, Optional

import paho.mqtt.client as mqtt

from . import config
from .logging_utils import log

MqttHandler = Callable[[dict[str, Any]], Awaitable[None]]


class MQTTBridge:
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        on_telemetry: MqttHandler,
        on_ack: MqttHandler,
        on_event: MqttHandler,
    ) -> None:
        self._host = host
        self._port = port
        self._on_telemetry = on_telemetry
        self._on_ack = on_ack
        self._on_event = on_event
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._inflight_count = 0
        self._inflight_lock = threading.Lock()

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if username:
            self._client.username_pw_set(username, password)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._client.connect(self._host, self._port, keepalive=config.MQTT_KEEPALIVE_S)
        self._client.loop_start()

    def stop(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def publish_command(self, payload: dict[str, Any]) -> bool:
        """Publish an ESP32-bound command. Returns True on successful enqueue."""
        try:
            payload_json = json.dumps(payload)
        except (TypeError, ValueError) as err:
            log("error", "command_serialize_failed", error=str(err))
            return False

        result = self._client.publish(
            config.MQTT_TOPIC_COMMANDS,
            payload_json,
            qos=config.MQTT_QOS_COMMAND,
        )
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            log("error", "mqtt_publish_failed", rc=result.rc)
            return False
        return True

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        # paho 2.x VERSION2: reason_code is a ReasonCode, not an int.
        if reason_code.is_failure:
            log(
                "error",
                "mqtt_connect_failed",
                reason_code=getattr(reason_code, "value", reason_code),
            )
            return
        log("info", "mqtt_connected", host=self._host)
        client.subscribe(
            [
                (config.MQTT_TOPIC_TELEMETRY, config.MQTT_QOS_TELEMETRY),
                (config.MQTT_TOPIC_ACK, config.MQTT_QOS_ACK),
                (config.MQTT_TOPIC_EVENTS, config.MQTT_QOS_TELEMETRY),
            ]
        )

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        # paho 2.x VERSION2: reason_code is a ReasonCode; never int() it here so a
        # logging/formatting error can't crash the disconnect handler.
        log(
            "warning",
            "mqtt_disconnected",
            reason_code=getattr(reason_code, "value", reason_code),
        )

    def _on_message(self, client, userdata, msg: mqtt.MQTTMessage) -> None:
        if self._loop is None or not self._loop.is_running():
            return

        with self._inflight_lock:
            if self._inflight_count >= config.MQTT_INFLIGHT_LIMIT:
                log(
                    "warning",
                    "mqtt_backpressure_drop",
                    topic=msg.topic,
                    inflight=self._inflight_count,
                )
                return
            self._inflight_count += 1

        try:
            data = json.loads(msg.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            log("warning", "mqtt_invalid_payload", topic=msg.topic, error=str(err))
            self._release_inflight_slot()
            return

        if not isinstance(data, dict):
            log("warning", "mqtt_payload_not_object", topic=msg.topic)
            self._release_inflight_slot()
            return

        handler = self._handler_for(msg.topic)
        if handler is None:
            log("warning", "mqtt_unknown_topic", topic=msg.topic)
            self._release_inflight_slot()
            return

        future = asyncio.run_coroutine_threadsafe(
            self._wrap_handler(handler, data), self._loop
        )
        future.add_done_callback(
            lambda fut: self._log_handler_exception(fut, msg.topic)
        )

    @staticmethod
    def _log_handler_exception(future: "asyncio.Future[None]", topic: str) -> None:
        """Surface (never swallow) exceptions raised by an inbound MQTT handler."""
        try:
            future.result()
        except Exception as err:
            log("error", "mqtt_handler_failed", topic=topic, error=str(err))

    async def _wrap_handler(self, handler: MqttHandler, data: dict[str, Any]) -> None:
        """Invoke the asyncio handler and always release the inflight slot."""
        try:
            await handler(data)
        finally:
            self._release_inflight_slot()

    def _release_inflight_slot(self) -> None:
        with self._inflight_lock:
            if self._inflight_count > 0:
                self._inflight_count -= 1

    def _handler_for(self, topic: str) -> Optional[MqttHandler]:
        if topic == config.MQTT_TOPIC_TELEMETRY:
            return self._on_telemetry
        if topic == config.MQTT_TOPIC_ACK:
            return self._on_ack
        if topic == config.MQTT_TOPIC_EVENTS:
            return self._on_event
        return None
