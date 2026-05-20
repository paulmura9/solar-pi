from __future__ import annotations

import asyncio
import json
import os
import random
import signal
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterator, Optional

import paho.mqtt.client as mqtt
import websockets
from dotenv import load_dotenv
from websockets.exceptions import ConnectionClosed, InvalidStatusCode


load_dotenv()


def _require_env(key: str) -> str:
    value = os.getenv(key, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable not set: {key}")
    return value


EXPRESS_WS_URL = _require_env("EXPRESS_WS_URL")
DEVICE_API_KEY = _require_env("DEVICE_API_KEY")
DEVICE_ID = os.getenv("DEVICE_ID", "raspberry-pi-001")

MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")

BUFFER_DB_PATH = Path(os.getenv("BUFFER_DB_PATH", "/var/lib/solar-tracker/buffer.db"))
BUFFER_MAX_AGE_HOURS = int(os.getenv("BUFFER_MAX_AGE_HOURS", "24"))

MQTT_TOPIC_TELEMETRY = "solar/telemetry"
MQTT_TOPIC_COMMANDS = "solar/commands"
MQTT_TOPIC_ACK = "solar/commands/ack"
MQTT_TOPIC_EVENTS = "solar/events"

MQTT_QOS_TELEMETRY = 1
MQTT_QOS_COMMAND = 2
MQTT_QOS_ACK = 1

MQTT_KEEPALIVE_S = 60
WS_HEARTBEAT_INTERVAL_S = 30
WS_PING_TIMEOUT_S = 10
WS_RECONNECT_MIN_DELAY_S = 1.0
WS_RECONNECT_MAX_DELAY_S = 60.0
WS_RECONNECT_JITTER_FACTOR = 0.3
BUFFER_FLUSH_BATCH_SIZE = 100
BUFFER_CLEANUP_INTERVAL_S = 3600
WS_SHUTDOWN_TIMEOUT_S = 5.0


def log(level: str, event: str, **fields: Any) -> None:
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        **fields,
    }
    print(json.dumps(entry, default=str), flush=True)


class OfflineBuffer:
    def __init__(self, db_path: Path, max_age_seconds: int) -> None:
        self._db_path = db_path
        self._max_age_seconds = max_age_seconds
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        try:
            yield conn
        finally:
            conn.close()

    def _initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS buffer (
                    id          TEXT PRIMARY KEY,
                    payload     TEXT NOT NULL,
                    created_at  REAL NOT NULL,
                    sent_at     REAL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_pending
                ON buffer(created_at)
                WHERE sent_at IS NULL
                """
            )
            conn.commit()

    async def enqueue(self, message_id: str, payload_json: str) -> None:
        await asyncio.to_thread(self._enqueue_sync, message_id, payload_json)

    def _enqueue_sync(self, message_id: str, payload_json: str) -> None:
        with self._connect() as conn:
            try:
                conn.execute(
                    "INSERT INTO buffer (id, payload, created_at) VALUES (?, ?, ?)",
                    (message_id, payload_json, time.time()),
                )
                conn.commit()
            except sqlite3.IntegrityError:
                pass

    async def get_pending(self, limit: int) -> list[tuple[str, str]]:
        return await asyncio.to_thread(self._get_pending_sync, limit)

    def _get_pending_sync(self, limit: int) -> list[tuple[str, str]]:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                SELECT id, payload FROM buffer
                WHERE sent_at IS NULL
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (limit,),
            )
            return [(row[0], row[1]) for row in cursor.fetchall()]

    async def mark_sent(self, message_id: str) -> None:
        await asyncio.to_thread(self._mark_sent_sync, message_id)

    def _mark_sent_sync(self, message_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE buffer SET sent_at = ? WHERE id = ?",
                (time.time(), message_id),
            )
            conn.commit()

    async def cleanup(self) -> int:
        return await asyncio.to_thread(self._cleanup_sync)

    def _cleanup_sync(self) -> int:
        now = time.time()
        pending_cutoff = now - self._max_age_seconds
        sent_cutoff = now - (7 * 24 * 3600)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM buffer
                WHERE (sent_at IS NULL AND created_at < ?)
                   OR (sent_at IS NOT NULL AND sent_at < ?)
                """,
                (pending_cutoff, sent_cutoff),
            )
            conn.commit()
            return cursor.rowcount


def validate_telemetry(data: dict[str, Any]) -> tuple[bool, str]:
    for field in ("horizontal_angle", "vertical_angle"):
        if field not in data:
            return False, f"missing field: {field}"

    h_angle = data["horizontal_angle"]
    v_angle = data["vertical_angle"]

    if not isinstance(h_angle, (int, float)) or not 0 <= h_angle <= 180:
        return False, f"horizontal_angle out of range: {h_angle}"

    if not isinstance(v_angle, (int, float)) or not 0 <= v_angle <= 180:
        return False, f"vertical_angle out of range: {v_angle}"

    optional_ranges = {
        "battery_voltage": (0, 15),
        "battery_percent": (0, 100),
        "solar_voltage": (0, 30),
        "solar_current": (0, 10),
        "solar_power": (0, 300),
    }

    for field_name, (low, high) in optional_ranges.items():
        value = data.get(field_name)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or not low <= value <= high:
            return False, f"{field_name} out of range: {value}"

    return True, "ok"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_envelope(message_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "v": 1,
        "type": message_type,
        "id": str(uuid.uuid4()),
        "timestamp": _now_iso(),
        "payload": payload,
    }


CommandHandler = Callable[[dict[str, Any]], Awaitable[None]]
MqttHandler = Callable[[dict[str, Any]], Awaitable[None]]


class WebSocketClient:
    def __init__(
        self,
        url: str,
        api_key: str,
        device_id: str,
        buffer: OfflineBuffer,
        on_command: CommandHandler,
    ) -> None:
        self._url = url
        self._headers = {"X-Device-Key": api_key, "X-Device-Id": device_id}
        self._buffer = buffer
        self._on_command = on_command
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._stop_event = asyncio.Event()
        self._last_command_id: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        return self._ws is not None

    def stop(self) -> None:
        self._stop_event.set()
        if self._ws is not None:
            asyncio.create_task(self._ws.close(code=1001, reason="shutdown"))

    async def run(self) -> None:
        backoff = WS_RECONNECT_MIN_DELAY_S

        while not self._stop_event.is_set():
            try:
                await self._connect_and_serve()
                backoff = WS_RECONNECT_MIN_DELAY_S
            except InvalidStatusCode as err:
                log("error", "ws_handshake_rejected", status=err.status_code)
            except (OSError, asyncio.TimeoutError, ConnectionClosed) as err:
                log("warning", "ws_connection_lost", error=str(err))
            except Exception as err:
                log("error", "ws_unexpected_error", error=str(err))

            if self._stop_event.is_set():
                break

            jitter = random.uniform(0, backoff * WS_RECONNECT_JITTER_FACTOR)
            delay = backoff + jitter
            log("info", "ws_reconnect_scheduled", delay_seconds=round(delay, 2))

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                break
            except asyncio.TimeoutError:
                pass

            backoff = min(backoff * 2, WS_RECONNECT_MAX_DELAY_S)

    async def send(self, message: dict[str, Any]) -> bool:
        if self._ws is None:
            return False
        try:
            await self._ws.send(json.dumps(message))
            return True
        except ConnectionClosed:
            return False

    async def send_or_buffer(self, message: dict[str, Any]) -> None:
        if await self.send(message):
            return

        message_id = message.get("id")
        if not isinstance(message_id, str):
            log("error", "buffer_missing_id", message_type=message.get("type"))
            return

        await self._buffer.enqueue(message_id, json.dumps(message))
        log("debug", "message_buffered", id=message_id, type=message.get("type"))

    async def _connect_and_serve(self) -> None:
        async with websockets.connect(
            self._url,
            additional_headers=self._headers,
            ping_interval=WS_HEARTBEAT_INTERVAL_S,
            ping_timeout=WS_PING_TIMEOUT_S,
            close_timeout=WS_SHUTDOWN_TIMEOUT_S,
            max_size=2**20,
        ) as ws:
            self._ws = ws
            log("info", "ws_connected", url=self._url)

            try:
                await self._send_sync_request()
                await self._flush_buffer()
                await self._receive_loop()
            finally:
                self._ws = None
                log("info", "ws_disconnected")

    async def _send_sync_request(self) -> None:
        await self._ws.send(  # type: ignore[union-attr]
            json.dumps(
                _build_envelope(
                    "sync_request",
                    {"last_command_id": self._last_command_id},
                )
            )
        )

    async def _flush_buffer(self) -> None:
        total_sent = 0
        while True:
            batch = await self._buffer.get_pending(BUFFER_FLUSH_BATCH_SIZE)
            if not batch:
                break

            for msg_id, payload_json in batch:
                if self._ws is None:
                    return
                try:
                    await self._ws.send(payload_json)
                    await self._buffer.mark_sent(msg_id)
                    total_sent += 1
                except ConnectionClosed:
                    log("warning", "buffer_flush_interrupted", flushed=total_sent)
                    return

        if total_sent > 0:
            log("info", "buffer_flushed", count=total_sent)

    async def _receive_loop(self) -> None:
        if self._ws is None:
            return
        async for raw in self._ws:
            await self._handle_incoming(raw)

    async def _handle_incoming(self, raw: str | bytes) -> None:
        try:
            text = raw if isinstance(raw, str) else raw.decode("utf-8")
            data = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            log("warning", "ws_invalid_json", error=str(err))
            return

        msg_type = data.get("type")

        if msg_type == "command":
            command_id = data.get("id")
            if not isinstance(command_id, str):
                log("warning", "command_missing_id")
                return
            self._last_command_id = command_id
            try:
                await self._on_command(data)
            except Exception as err:
                log("error", "command_handler_failed", id=command_id, error=str(err))

        elif msg_type == "heartbeat_ack":
            pass

        elif msg_type == "server_shutting_down":
            log("warning", "ws_server_shutdown_notice")

        else:
            log("warning", "ws_unknown_message_type", type=msg_type)


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

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if username:
            self._client.username_pw_set(username, password)
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._client.connect(self._host, self._port, keepalive=MQTT_KEEPALIVE_S)
        self._client.loop_start()

    def stop(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def publish_command(self, payload: dict[str, Any]) -> bool:
        try:
            payload_json = json.dumps(payload)
        except (TypeError, ValueError) as err:
            log("error", "command_serialize_failed", error=str(err))
            return False

        result = self._client.publish(
            MQTT_TOPIC_COMMANDS,
            payload_json,
            qos=MQTT_QOS_COMMAND,
        )
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            log("error", "mqtt_publish_failed", rc=result.rc)
            return False
        return True

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code == 0:
            log("info", "mqtt_connected", host=self._host)
            client.subscribe(
                [
                    (MQTT_TOPIC_TELEMETRY, MQTT_QOS_TELEMETRY),
                    (MQTT_TOPIC_ACK, MQTT_QOS_ACK),
                    (MQTT_TOPIC_EVENTS, MQTT_QOS_TELEMETRY),
                ]
            )
        else:
            log("error", "mqtt_connect_failed", reason_code=int(reason_code))

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        log("warning", "mqtt_disconnected", reason_code=int(reason_code))

    def _on_message(self, client, userdata, msg: mqtt.MQTTMessage) -> None:
        if self._loop is None or not self._loop.is_running():
            return

        try:
            data = json.loads(msg.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            log("warning", "mqtt_invalid_payload", topic=msg.topic, error=str(err))
            return

        if not isinstance(data, dict):
            log("warning", "mqtt_payload_not_object", topic=msg.topic)
            return

        handler = self._handler_for(msg.topic)
        if handler is None:
            log("warning", "mqtt_unknown_topic", topic=msg.topic)
            return

        asyncio.run_coroutine_threadsafe(handler(data), self._loop)

    def _handler_for(self, topic: str) -> Optional[MqttHandler]:
        if topic == MQTT_TOPIC_TELEMETRY:
            return self._on_telemetry
        if topic == MQTT_TOPIC_ACK:
            return self._on_ack
        if topic == MQTT_TOPIC_EVENTS:
            return self._on_event
        return None


class SolarGateway:
    def __init__(self) -> None:
        self._buffer = OfflineBuffer(
            db_path=BUFFER_DB_PATH,
            max_age_seconds=BUFFER_MAX_AGE_HOURS * 3600,
        )
        self._ws = WebSocketClient(
            url=EXPRESS_WS_URL,
            api_key=DEVICE_API_KEY,
            device_id=DEVICE_ID,
            buffer=self._buffer,
            on_command=self._on_command_from_cloud,
        )
        self._mqtt = MQTTBridge(
            host=MQTT_BROKER_HOST,
            port=MQTT_BROKER_PORT,
            username=MQTT_USERNAME,
            password=MQTT_PASSWORD,
            on_telemetry=self._forward_telemetry,
            on_ack=self._forward_ack,
            on_event=self._forward_event,
        )
        self._stop_event = asyncio.Event()

    def stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        log("info", "gateway_starting", device_id=DEVICE_ID, ws_url=EXPRESS_WS_URL)

        loop = asyncio.get_running_loop()
        self._mqtt.start(loop)

        ws_task = asyncio.create_task(self._ws.run())
        heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        cleanup_task = asyncio.create_task(self._cleanup_loop())

        await self._stop_event.wait()

        log("info", "gateway_stopping")

        self._ws.stop()
        heartbeat_task.cancel()
        cleanup_task.cancel()

        try:
            await asyncio.wait_for(ws_task, timeout=WS_SHUTDOWN_TIMEOUT_S)
        except asyncio.TimeoutError:
            log("warning", "ws_shutdown_timeout")

        self._mqtt.stop()
        log("info", "gateway_stopped")

    async def _forward_telemetry(self, data: dict[str, Any]) -> None:
        valid, reason = validate_telemetry(data)
        if not valid:
            log("warning", "telemetry_rejected", reason=reason)
            return

        message = _build_envelope("telemetry", data)
        await self._ws.send_or_buffer(message)

    async def _forward_ack(self, data: dict[str, Any]) -> None:
        command_id = data.get("commandId")
        status = data.get("status")

        if not command_id or not status:
            log("warning", "ack_incomplete", data=data)
            return

        message = _build_envelope("command_ack", data)
        await self._ws.send_or_buffer(message)
        log("info", "command_acked", command_id=command_id, status=status)

    async def _forward_event(self, data: dict[str, Any]) -> None:
        message = _build_envelope("esp32_event", data)
        await self._ws.send_or_buffer(message)

    async def _on_command_from_cloud(self, message: dict[str, Any]) -> None:
        command_id = message["id"]
        inner = message.get("payload", {})
        command_type = inner.get("command_type")
        args = inner.get("args", {})

        log("info", "command_received", id=command_id, type=command_type)

        mqtt_payload = {
            "commandId": command_id,
            "command_type": command_type,
            "payload": args,
        }

        success = self._mqtt.publish_command(mqtt_payload)

        if not success:
            failure_ack = _build_envelope(
                "command_ack",
                {
                    "commandId": command_id,
                    "status": "FAILED",
                    "message": "MQTT publish failed on gateway",
                },
            )
            await self._ws.send_or_buffer(failure_ack)

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(WS_HEARTBEAT_INTERVAL_S)
                if self._ws.is_connected:
                    await self._ws.send(_build_envelope("heartbeat", {}))
        except asyncio.CancelledError:
            pass

    async def _cleanup_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(BUFFER_CLEANUP_INTERVAL_S)
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


if __name__ == "__main__":
    asyncio.run(main())