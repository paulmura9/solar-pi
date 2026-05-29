"""Structured JSON logging.

Single responsibility: emit one-line JSON records to stdout so systemd's journal
captures machine-parseable logs. Centralized here so every module logs uniformly.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def log(level: str, event: str, **fields: Any) -> None:
    """Emit a structured log record as a single JSON line on stdout."""
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        **fields,
    }
    print(json.dumps(entry, default=str), flush=True)
