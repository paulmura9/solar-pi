"""Permissive Pi-side telemetry validation.

Required fields (the servo angles) are bounds-checked to catch obvious firmware
bugs early; optional fields with known sane ranges are checked only when present.
Strict schema validation is the backend's responsibility (the Pi is a forwarder,
not the schema authority), so re-validating the full schema here is deliberately
avoided to prevent a second, drift-prone source of truth.
"""
from __future__ import annotations

from typing import Any

from . import config

ANGLE_MIN_DEG = 0
ANGLE_MAX_DEG = 180
REQUIRED_ANGLE_FIELDS = ("horizontal_angle", "vertical_angle")

TELEMETRY_OPTIONAL_RANGES = {
    "battery_voltage": (0, 15),
    "battery_percent": (0, 100),
    "solar_voltage": (0, 30),
}

TELEMETRY_CLAMP_RANGES = {
    "solar_current": (config.SOLAR_CURRENT_MIN_A, config.SOLAR_CURRENT_MAX_A),
    "solar_power": (config.SOLAR_POWER_MIN_W, config.SOLAR_POWER_MAX_W),
}

TELEMETRY_NONNEGATIVE_MINIMUMS = {
    "solar_energy_today_wh": config.SOLAR_ENERGY_TODAY_MIN_WH,
}


def validate_telemetry(data: dict[str, Any]) -> tuple[bool, str]:
    """Validate telemetry, clamping near-zero sensor noise in-place.

    Returns (is_valid, reason); reason is "ok" when valid. solar_current and
    solar_power are clamped (not rejected) when a small-negative noise reading
    falls within their negative deadband, so angles/LDR/battery are preserved.
    """
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

    for field_name, (low, high) in TELEMETRY_CLAMP_RANGES.items():
        value = data.get(field_name)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or not low <= value <= high:
            return False, f"{field_name} out of range: {value}"
        if value < config.TELEMETRY_NOISE_CLAMP_FLOOR:
            data[field_name] = config.TELEMETRY_NOISE_CLAMP_FLOOR

    for field_name, minimum in TELEMETRY_NONNEGATIVE_MINIMUMS.items():
        value = data.get(field_name)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or value < minimum:
            return False, f"{field_name} out of range: {value}"

    return True, "ok"
