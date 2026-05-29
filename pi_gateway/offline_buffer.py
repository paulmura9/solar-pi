"""SQLite-backed offline buffer for outbound messages.

When the WebSocket to Express is down, outbound messages are persisted here and
flushed once the connection returns, giving the gateway offline resilience.
All SQLite calls run in a worker thread so they never block the event loop.
"""
from __future__ import annotations

import asyncio
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from . import config


class OfflineBuffer:
    def __init__(self, db_path: Path, max_age_seconds: int) -> None:
        self._db_path = db_path
        self._max_age_seconds = max_age_seconds
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, timeout=config.SQLITE_BUSY_TIMEOUT_S)
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
                # Duplicate id: the message is already buffered; nothing to do.
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
        sent_cutoff = now - config.BUFFER_SENT_RETENTION_S
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
