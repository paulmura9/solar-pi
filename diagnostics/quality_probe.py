"""Temporary diagnostic: measure the quality-gate metrics on live frames.

NOT part of the gateway. Run it by hand on the Pi to calibrate the obstruction
quality gate in pi_gateway/vision.py (check_frame_quality). It captures a frame,
applies the SAME ROI crop as the model, and prints the grayscale mean and std,
the per-channel BGR means, the red-over-blue dominance, and the verdict the gate
would give. Take readings with a finger on the lens, a covered lens, darkness,
direct light, and a normal panel to pick good thresholds.

It owns the camera only for the session and touches nothing else (no MQTT, WS,
Supabase, model). To avoid pulling in the model runtime and the required-env
config import, the ROI and thresholds are duplicated here as local constants:
keep them in sync with pi_gateway/config.py while calibrating.

    python3 diagnostics/quality_probe.py
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator, Optional

import cv2
from picamera2 import Picamera2

# --- Mirrors of pi_gateway/config.py (keep in sync) ---------------------------

# Full-frame capture size and pixel format, matching the gateway/dataset tool:
# picamera2 yields RGB-ordered arrays on this Pi, converted to BGR for OpenCV.
CAPTURE_SIZE = (2304, 1296)
CAMERA_PIXEL_FORMAT = "BGR888"

# Fixed panel ROI (x, y, w, h) in full-frame pixels (config.DIRT_ROI_*).
ROI_X, ROI_Y, ROI_W, ROI_H = 440, 40, 1850, 1220

# Quality-gate thresholds (config.DIRT_QUALITY_*). Verdict order matches
# check_frame_quality: too_dark -> too_bright -> not_panel -> low_detail.
QUALITY_MIN_MEAN = 30
QUALITY_MAX_MEAN = 220
QUALITY_RED_DOMINANCE = 25
QUALITY_MIN_STD = 15

REASON_TOO_DARK = "too_dark"
REASON_TOO_BRIGHT = "too_bright"
REASON_NOT_PANEL = "not_panel"
REASON_LOW_DETAIL = "low_detail"


@contextmanager
def _open_camera() -> Iterator[Picamera2]:
    """Start the camera at CAPTURE_SIZE; always release it on exit."""
    try:
        camera = Picamera2()
    except Exception as exc:
        raise SystemExit(f"camera unavailable: {exc}") from exc
    try:
        camera.configure(
            camera.create_still_configuration(
                main={"size": CAPTURE_SIZE, "format": CAMERA_PIXEL_FORMAT}
            )
        )
        camera.start()
    except Exception as exc:
        camera.close()
        raise SystemExit(f"camera start failed: {exc}") from exc

    print(f"camera started at {CAPTURE_SIZE[0]}x{CAPTURE_SIZE[1]}")
    try:
        yield camera
    finally:
        try:
            camera.stop()
        finally:
            camera.close()
            print("camera stopped.")


def _crop_to_roi(frame_bgr):
    """Crop to the fixed ROI; fall back to the full frame if it does not fit."""
    height, width = frame_bgr.shape[0], frame_bgr.shape[1]
    if ROI_X < 0 or ROI_Y < 0 or ROI_X + ROI_W > width or ROI_Y + ROI_H > height:
        print(f"  WARNING: ROI ({ROI_X},{ROI_Y},{ROI_W},{ROI_H}) does not fit "
              f"{width}x{height}; using full frame (matches gate fallback).")
        return frame_bgr
    return frame_bgr[ROI_Y : ROI_Y + ROI_H, ROI_X : ROI_X + ROI_W]


def _verdict(mean: float, std: float, mean_b: float, mean_r: float) -> Optional[str]:
    """Same conditions/order as vision.check_frame_quality."""
    if mean < QUALITY_MIN_MEAN:
        return REASON_TOO_DARK
    if mean > QUALITY_MAX_MEAN:
        return REASON_TOO_BRIGHT
    if mean_r - mean_b > QUALITY_RED_DOMINANCE:
        return REASON_NOT_PANEL
    if std < QUALITY_MIN_STD:
        return REASON_LOW_DETAIL
    return None


def _measure(camera: Picamera2) -> None:
    frame = camera.capture_array()
    if frame is None or frame.size == 0:
        print("  capture returned an empty frame.")
        return
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    crop = _crop_to_roi(frame_bgr)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    mean, std = float(gray.mean()), float(gray.std())
    mean_b, mean_g, mean_r = (float(channel) for channel in cv2.mean(crop)[:3])
    red_dominance = mean_r - mean_b

    reason = _verdict(mean, std, mean_b, mean_r)
    status = "OBSTRUCTED -> " + reason if reason else "OK (would run the model)"
    print(
        f"  gray mean={mean:6.2f} (dark<{QUALITY_MIN_MEAN}, bright>{QUALITY_MAX_MEAN})  "
        f"std={std:6.2f} (detail<{QUALITY_MIN_STD})\n"
        f"  B={mean_b:6.2f}  G={mean_g:6.2f}  R={mean_r:6.2f}  "
        f"R-B={red_dominance:+6.2f} (not_panel>{QUALITY_RED_DOMINANCE})\n"
        f"  ->  {status}"
    )


def main() -> None:
    print("Quality-gate probe. Press ENTER to measure a frame, 'q' then ENTER to quit.")
    print("Try: finger on lens, covered lens, darkness, direct light, normal panel.\n")
    try:
        with _open_camera() as camera:
            while True:
                command = input("[ENTER]=measure  q=quit > ").strip().lower()
                if command == "q":
                    break
                if command != "":
                    print("  press ENTER to measure or 'q' to quit.")
                    continue
                try:
                    _measure(camera)
                except Exception as exc:
                    print(f"  measurement failed: {exc}")
    except KeyboardInterrupt:
        print("\ninterrupted.")
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
