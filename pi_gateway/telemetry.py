"""Permissive Pi-side telemetry validation.

Required fields (the servo angles) are bounds-checked to catch obvious firmware
bugs early; optional fields with known sane ranges are checked only when present.
Strict schema validation is the backend's responsibility (the Pi is a forwarder,
not the schema authority), so re-validating the full schema here is deliberately
avoided to prevent a second, drift-prone source of truth.
"""
from __future__ import annotations

from typing import Any

# Servo angle bounds (degrees); the ESP32 mechanism spans this range.
ANGLE_MIN_DEG = 0
ANGLE_MAX_DEG = 180
REQUIRED_ANGLE_FIELDS = ("horizontal_angle", "vertical_angle")

# Optional sensor fields with known-sane (low, high) ranges, checked when present.
TELEMETRY_OPTIONAL_RANGES = {
    "battery_voltage": (0, 15),
    "battery_percent": (0, 100),
    "solar_voltage": (0, 30),
    "solar_current": (0, 10),
    "solar_power": (0, 300),
}


def validate_telemetry(data: dict[str, Any]) -> tuple[bool, str]:
    """Return (is_valid, reason). Reason is "ok" when valid."""
    for field in REQUIRED_ANGLE_FIELDS:
        if field not in data:
            return False, f"missing field: {field}"

    for field in REQUIRED_ANGLE_FIELDS:
        value = data[field]
        if not isinstance(value, (int, float)) or not ANGLE_MIN_DEG <= value <= ANGLE_MAX_DEG:
            return False, f"{field} out of range: {value}"

    for field_name, (low, high) in TELEMETRY_OPTIONAL_RANGES.items():
        value = data.get(field_name)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or not low <= value <= high:
            return False, f"{field_name} out of range: {value}"

    return True, "ok"
