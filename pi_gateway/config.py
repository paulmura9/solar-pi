"""Centralized configuration and named constants.

All tunables and secrets live here, loaded from the environment (never hardcoded)
or expressed as named constants (no magic numbers scattered across the code).
Required secrets fail fast at import time so misconfiguration is caught at startup.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _require_env(key: str) -> str:
    value = os.getenv(key, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable not set: {key}")
    return value


# --- Express control plane (Pi -> Express, persistent WebSocket client) -------
# EXPRESS_WS_URL must target the backend's /ws/device endpoint.
EXPRESS_WS_URL = _require_env("EXPRESS_WS_URL")
DEVICE_API_KEY = _require_env("DEVICE_API_KEY")
DEVICE_ID = os.getenv("DEVICE_ID", "raspberry-pi-001")

# --- Supabase Storage (Pi -> Storage over HTTPS; the Pi has NO database access) -
SUPABASE_URL = _require_env("SUPABASE_URL")
SUPABASE_STORAGE_KEY = _require_env("SUPABASE_STORAGE_KEY")
# Same bucket already used for vision images. Provided via env so the bucket name
# is never hardcoded or guessed in source.
SUPABASE_STORAGE_BUCKET = _require_env("SUPABASE_STORAGE_BUCKET")

# --- Local MQTT broker (Pi <-> ESP32, never exposed publicly) -----------------
MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")

# --- Offline buffer (SQLite) --------------------------------------------------
BUFFER_DB_PATH = Path(os.getenv("BUFFER_DB_PATH", "/var/lib/solar-tracker/buffer.db"))
BUFFER_MAX_AGE_HOURS = int(os.getenv("BUFFER_MAX_AGE_HOURS", "24"))
# How long a SQLite connection waits for a competing writer to release its lock
# before raising "database is locked".
SQLITE_BUSY_TIMEOUT_S = 5.0

# --- MQTT topics / QoS (established contract with the ESP32 firmware) ----------
MQTT_TOPIC_TELEMETRY = "solar/telemetry"
MQTT_TOPIC_COMMANDS = "solar/commands"
MQTT_TOPIC_ACK = "solar/commands/ack"
MQTT_TOPIC_EVENTS = "solar/events"

MQTT_QOS_TELEMETRY = 1
MQTT_QOS_COMMAND = 2
MQTT_QOS_ACK = 1

MQTT_KEEPALIVE_S = 60

# Max MQTT messages queued for asyncio processing at once. Beyond this, incoming
# MQTT messages are dropped to bound memory when the WebSocket is down and the
# offline buffer cannot keep up.
MQTT_INFLIGHT_LIMIT = 1000

# --- WebSocket timing / reconnect backoff schedule ----------------------------
WS_HEARTBEAT_INTERVAL_S = 30
WS_PING_TIMEOUT_S = 10
WS_RECONNECT_MIN_DELAY_S = 1.0
WS_RECONNECT_MAX_DELAY_S = 60.0
WS_RECONNECT_JITTER_FACTOR = 0.3
WS_RECONNECT_BACKOFF_FACTOR = 2
WS_SHUTDOWN_TIMEOUT_S = 5.0
WS_MAX_MESSAGE_BYTES = 2 ** 20

# --- Offline buffer flush / cleanup -------------------------------------------
BUFFER_FLUSH_BATCH_SIZE = 100
BUFFER_CLEANUP_INTERVAL_S = 3600
BUFFER_SENT_RETENTION_S = 7 * 24 * 3600

# --- Camera / capture ---------------------------------------------------------
# IMX708 full-frame size verified in scripts/camera_test.py. picamera2's format
# string does not reliably match numpy channel order on the Pi; the array comes
# out RGB-ordered, so CameraManager converts to BGR for OpenCV.
CAMERA_RESOLUTION = (2304, 1296)
CAMERA_PIXEL_FORMAT = "BGR888"
CAPTURE_JPEG_QUALITY = 90
CAPTURE_TIMEOUT_S = 15.0
# Max time to wait for the CameraManager lock before reporting the camera busy.
# Kept >= CAPTURE_TIMEOUT_S so a legitimately in-progress capture is allowed to
# finish (or hit its own timeout) rather than being pre-empted as "busy"; it
# also bounds the wait so a wedged capture can never block callers forever.
CAMERA_LOCK_TIMEOUT_S = 20.0

# Storage object-key prefix for manual captures: captures/<timestamp>_<id>.jpg
STORAGE_CAPTURE_PREFIX = "captures"

# --- Telemetry sensor validation (SI units) -----------------------------------
# solar_current (Amperes) and solar_power (Watts) are clamp-then-validate: a
# small-negative reading (sensor noise near zero) is clamped to the noise floor
# instead of dropping the whole packet; only readings outside [MIN, MAX] are
# rejected. The panel is ~0.35 A / ~4.2 W rated, so the maxima leave headroom.
TELEMETRY_NOISE_CLAMP_FLOOR = 0.0
SOLAR_CURRENT_MIN_A = -0.5
SOLAR_CURRENT_MAX_A = 2.0
SOLAR_POWER_MIN_W = -0.5
SOLAR_POWER_MAX_W = 5.0
# Daily energy accumulator (Wh). Non-negative with no fixed upper bound, since it
# grows over the day and resets externally; only negatives are rejected.
SOLAR_ENERGY_TODAY_MIN_WH = 0.0
