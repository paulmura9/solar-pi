"""Persistent WebSocket client to the Express backend (/ws/device).

Responsibilities:
  - Maintain one authenticated, persistent connection with exponential-backoff
    reconnect (schedule from config).
  - Deliver inbound "command" messages to the injected dispatcher callback.
  - Provide send()/send_or_buffer() for the rest of the gateway to report results.

It owns no business logic: command routing lives in the dispatcher, persistence
in the offline buffer.
"""
from __future__ import annotations

import asyncio
import json
import random
from typing import Any, Awaitable, Callable, Optional

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

from . import config, protocol
from .logging_utils import log
from .offline_buffer import OfflineBuffer

CommandHandler = Callable[[dict[str, Any]], Awaitable[None]]


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
        """Signal the run loop to exit on the next iteration.

        The WebSocket close happens naturally when `_receive_loop` returns and
        the `async with` block in `_connect_and_serve` exits. We avoid creating
        a task here because `stop()` may be called from a signal handler.
        """
        self._stop_event.set()

    async def run(self) -> None:
        backoff = config.WS_RECONNECT_MIN_DELAY_S

        while not self._stop_event.is_set():
            try:
                await self._connect_and_serve()
                backoff = config.WS_RECONNECT_MIN_DELAY_S
            except InvalidStatus as err:
                log("error", "ws_handshake_rejected", status=err.response.status_code)
            except (OSError, asyncio.TimeoutError, ConnectionClosed) as err:
                log("warning", "ws_connection_lost", error=str(err))
            except asyncio.CancelledError:
                raise
            except Exception as err:
                log("error", "ws_unexpected_error", error=str(err))

            if self._stop_event.is_set():
                break

            jitter = random.uniform(0, backoff * config.WS_RECONNECT_JITTER_FACTOR)
            delay = backoff + jitter
            log("info", "ws_reconnect_scheduled", delay_seconds=round(delay, 2))

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
                break
            except asyncio.TimeoutError:
                pass

            backoff = min(
                backoff * config.WS_RECONNECT_BACKOFF_FACTOR,
                config.WS_RECONNECT_MAX_DELAY_S,
            )

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
            ping_interval=config.WS_HEARTBEAT_INTERVAL_S,
            ping_timeout=config.WS_PING_TIMEOUT_S,
            close_timeout=config.WS_SHUTDOWN_TIMEOUT_S,
            max_size=config.WS_MAX_MESSAGE_BYTES,
        ) as ws:
            self._ws = ws
            log("info", "ws_connected", url=self._url)

            flush_task: Optional[asyncio.Task[None]] = None
            try:
                await self._send_sync_request()
                # Flush the buffer in the background so command reception is not
                # blocked by a large drain after a long outage.
                flush_task = asyncio.create_task(self._flush_buffer())
                await self._receive_loop()
            finally:
                if flush_task is not None and not flush_task.done():
                    flush_task.cancel()
                    try:
                        await flush_task
                    except asyncio.CancelledError:
                        pass
                self._ws = None
                log("info", "ws_disconnected")

    async def _send_sync_request(self) -> None:
        """Tell the backend the last command_id processed before this connection.

        `_last_command_id` is in-memory only. A Pi restart resets it to None and
        the backend will not replay older commands (their PENDING timeout has
        elapsed) - acceptable, since replaying stale motor commands is unsafe.
        """
        await self.send(
            protocol.build_envelope(
                protocol.MSG_TYPE_SYNC_REQUEST,
                {"last_command_id": self._last_command_id},
            )
        )

    async def _flush_buffer(self) -> None:
        total_sent = 0
        try:
            while True:
                batch = await self._buffer.get_pending(config.BUFFER_FLUSH_BATCH_SIZE)
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
        except asyncio.CancelledError:
            log("info", "buffer_flush_cancelled", flushed=total_sent)
            raise
        finally:
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

        if msg_type == protocol.MSG_TYPE_COMMAND:
            command_id = data.get("id")
            if not isinstance(command_id, str):
                log("warning", "command_missing_id")
                return
            self._last_command_id = command_id
            try:
                await self._on_command(data)
            except Exception as err:
                log("error", "command_handler_failed", id=command_id, error=str(err))

        elif msg_type == protocol.MSG_TYPE_HEARTBEAT_ACK:
            pass

        elif msg_type == protocol.MSG_TYPE_SERVER_SHUTDOWN:
            log("warning", "ws_server_shutdown_notice")

        else:
            log("warning", "ws_unknown_message_type", type=msg_type)
