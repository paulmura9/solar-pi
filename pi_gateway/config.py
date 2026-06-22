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


EXPRESS_WS_URL = _require_env("EXPRESS_WS_URL")
DEVICE_API_KEY = _require_env("DEVICE_API_KEY")
DEVICE_ID = os.getenv("DEVICE_ID", "raspberry-pi-001")

SUPABASE_URL = _require_env("SUPABASE_URL")
SUPABASE_STORAGE_KEY = _require_env("SUPABASE_STORAGE_KEY")
SUPABASE_STORAGE_BUCKET = _require_env("SUPABASE_STORAGE_BUCKET")

MQTT_BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_BROKER_PORT = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME", "")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")

BUFFER_DB_PATH = Path(os.getenv("BUFFER_DB_PATH", "/var/lib/solar-tracker/buffer.db"))
BUFFER_MAX_AGE_HOURS = int(os.getenv("BUFFER_MAX_AGE_HOURS", "24"))
SQLITE_BUSY_TIMEOUT_S = 5.0

MQTT_TOPIC_TELEMETRY = "solar/telemetry"
MQTT_TOPIC_COMMANDS = "solar/commands"
MQTT_TOPIC_ACK = "solar/commands/ack"
MQTT_TOPIC_EVENTS = "solar/events"

MQTT_QOS_TELEMETRY = 1
MQTT_QOS_COMMAND = 2
MQTT_QOS_ACK = 1

MQTT_KEEPALIVE_S = 60

MQTT_INFLIGHT_LIMIT = 1000

WS_HEARTBEAT_INTERVAL_S = 30
WS_PING_TIMEOUT_S = 10
WS_RECONNECT_MIN_DELAY_S = 1.0
WS_RECONNECT_MAX_DELAY_S = 60.0
WS_RECONNECT_JITTER_FACTOR = 0.3
WS_RECONNECT_BACKOFF_FACTOR = 2
WS_SHUTDOWN_TIMEOUT_S = 5.0
WS_MAX_MESSAGE_BYTES = 2 ** 20

BUFFER_FLUSH_BATCH_SIZE = 100
BUFFER_CLEANUP_INTERVAL_S = 3600
BUFFER_SENT_RETENTION_S = 7 * 24 * 3600

CAMERA_RESOLUTION = (2304, 1296)
CAMERA_PIXEL_FORMAT = "BGR888"
CAPTURE_JPEG_QUALITY = 90
CAPTURE_TIMEOUT_S = 15.0
CAMERA_LOCK_TIMEOUT_S = 20.0

STORAGE_CAPTURE_PREFIX = "captures"

VISION_ENABLED = os.getenv("VISION_ENABLED", "true").strip().lower() not in (
    "false",
    "0",
    "no",
    "off",
)

VISION_CAPTURE_INTERVAL_S = int(os.getenv("VISION_CAPTURE_INTERVAL_S", "1800"))
DIRT_MODEL_PATH = os.getenv("DIRT_MODEL_PATH", "models/dirt_detection.tflite")
STORAGE_VISION_PREFIX = "vision"
DIRT_ROI_X = 440
DIRT_ROI_Y = 40
DIRT_ROI_W = 1850
DIRT_ROI_H = 1220

DIRT_QUALITY_MIN_MEAN = 30
DIRT_QUALITY_MAX_MEAN = 220
DIRT_QUALITY_RED_DOMINANCE = 25
DIRT_QUALITY_MIN_STD = 15

SURFACE_BLUR_KERNEL = 51
SURFACE_DIFF_THRESHOLD = 25
SURFACE_HIGHLIGHT_COLOR_BGR = (0, 0, 255)
SURFACE_OVERLAY_ALPHA = 0.5
SURFACE_HLINE_LENGTH = 25
SURFACE_VLINE_LENGTH = 25
STORAGE_SURFACE_PREFIX = "surface"

TELEMETRY_NOISE_CLAMP_FLOOR = 0.0
SOLAR_CURRENT_MIN_A = -0.5
SOLAR_CURRENT_MAX_A = 2.0
SOLAR_POWER_MIN_W = -0.5
SOLAR_POWER_MAX_W = 5.0
SOLAR_ENERGY_TODAY_MIN_WH = 0.0
