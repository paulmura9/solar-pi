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

# Vision subsystem toggle. When false, the gateway never opens the camera (no
# periodic capture, no stream), leaving the exclusive Pi camera free for the
# offline dataset-collection script (dataset/capture_dataset.py). Every other
# path is unaffected: WebSocket, MQTT bridge, command forwarding to the ESP32,
# telemetry, ACKs, heartbeat. Fail-safe default: anything other than an explicit
# off-value keeps vision enabled, so a typo never silently disables the camera.
VISION_ENABLED = os.getenv("VISION_ENABLED", "true").strip().lower() not in (
    "false",
    "0",
    "no",
    "off",
)

# --- Vision / dirt detection --------------------------------------------------
# How often the periodic edge dirt-detection loop captures and classifies a
# frame. 30 min default keeps the camera mostly free and Storage writes modest.
VISION_CAPTURE_INTERVAL_S = int(os.getenv("VISION_CAPTURE_INTERVAL_S", "1800"))
# Trained TFLite model location (same env var as inference/predict_dirt.py).
DIRT_MODEL_PATH = os.getenv("DIRT_MODEL_PATH", "models/dirt_detection.tflite")
# Storage object-key prefix for vision frames (reuses SUPABASE_STORAGE_BUCKET):
# vision/<timestamp>.jpg. Analogous to STORAGE_CAPTURE_PREFIX.
STORAGE_VISION_PREFIX = "vision"
# Fixed panel ROI cropped before resize, in full-frame (CAMERA_RESOLUTION,
# 2304x1296) pixel coordinates: (x, y, width, height). The model was retrained on
# images cropped to this ROI, so inference MUST apply the identical crop or the
# predictions are invalid. Must match the crop used in the Colab training set.
DIRT_ROI_X = 440
DIRT_ROI_Y = 40
DIRT_ROI_W = 1850
DIRT_ROI_H = 1220

# Pre-inference quality gate, computed on the ROI crop's grayscale mean/std, to
# reject obstructed frames (covered lens, a hand over it, darkness) before they
# reach the model. Heuristics - calibrate on real captures.
# Below this average brightness (0..255) the frame is treated as too dark.
DIRT_QUALITY_MIN_MEAN = 30
# Above this average brightness the frame is overexposed/washed out (direct light
# into the lens, a white object).
DIRT_QUALITY_MAX_MEAN = 220
# Colour gate: the panel is dark blue (mean_B >= mean_R). When red dominates the
# blue channel by more than this (BGR means), the frame is likely skin/an object,
# not the panel.
DIRT_QUALITY_RED_DOMINANCE = 25
# Below this contrast (grayscale std) the frame has too little detail to analyze
# (e.g. a uniform surface covering the lens).
DIRT_QUALITY_MIN_STD = 15

# --- Surface analysis overlay (classical OpenCV, independent of the ML model) -
# Auxiliary visual aid shipped as vision_result.processed_image_path: it estimates
# surface deposits by image processing (local highlights brighter than the panel),
# NOT what the CNN sees. Class/percentages still come solely from the model.
# Gaussian blur kernel (must be odd) used to estimate the uneven background
# lighting; a large kernel keeps real deposits while smoothing illumination.
SURFACE_BLUR_KERNEL = 51
# A pixel is flagged as a deposit when it is at least this much brighter (0..255)
# than the local background estimate.
SURFACE_DIFF_THRESHOLD = 25
# Highlight colour (BGR) and opacity (0..1) of the deposit overlay on the crop.
SURFACE_HIGHLIGHT_COLOR_BGR = (0, 0, 255)
SURFACE_OVERLAY_ALPHA = 0.5
# Bus-bar removal by direction-aware morphology. In the straightened panel the bus
# bars run strictly horizontal and the cell edges strictly vertical, while dirt is
# isotropic. A morphological opening with a long thin LINE kernel keeps only the runs
# at least this many pixels long in that direction (the bus bars / cell edges), which
# are then subtracted from the dirt mask. Compact deposits match neither line kernel
# and survive regardless of size - unlike a uniform opening, which erases the small
# specks before the thicker bus bars. Bus bars span the whole panel, so these can be
# raised to spare genuinely elongated dirt; lower values catch shorter line fragments.
SURFACE_HLINE_LENGTH = 25  # horizontal-line kernel length (px), for the bus bars
SURFACE_VLINE_LENGTH = 25  # vertical-line kernel length (px), for the cell edges
# Storage object-key prefix for the overlay images (reuses SUPABASE_STORAGE_BUCKET).
STORAGE_SURFACE_PREFIX = "surface"

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
