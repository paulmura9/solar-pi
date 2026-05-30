"""Interactive dataset-collection tool for the dirt-detection model.

Standalone utility, NOT part of the production gateway. Run it by hand on the
Raspberry Pi to build a labelled image dataset for training/evaluation. It does
not touch MQTT, the WebSocket, Supabase, or any gateway module: it only owns the
camera for the duration of a collection session and writes files under dataset/.

Workflow
--------
1. The operator enters a free-form ``session_id`` and picks one class.
2. Each ENTER captures one full-frame still and saves it; 'q' ends the session.
   Capturing one frame at a time (instead of a burst) lets the operator
   rearrange the dust on the panel between shots.

Design choices that match the gateway's verified hardware behaviour
-------------------------------------------------------------------
- Full field of view, no crop, captured at ``CAPTURE_SIZE``. The Pi 3B cannot
  allocate the IMX708 full sensor readout (4608x2592 -> ENOMEM at camera
  start), so the 2x2-binned 2304x1296 mode is used: same whole-frame FOV, lower
  resolution. The ROI is applied later in preprocessing, not here.
- picamera2's format string does not reliably match numpy channel order on the
  Pi: the array comes out RGB-ordered, so it is converted to BGR before OpenCV
  encodes it (same reasoning as pi_gateway/camera_manager.py).
"""
from __future__ import annotations

import csv
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import cv2
from picamera2 import Picamera2

# --- Named constants (no magic numbers) ---------------------------------------

# The three labels the dirt-detection dataset is built from.
VALID_CLASSES = ("clean", "slightly_dirty", "dirty")

# picamera2 yields RGB-ordered arrays on this Pi; OpenCV needs BGR for encoding.
CAMERA_PIXEL_FORMAT = "BGR888"

# Capture size. The Pi 3B cannot allocate buffers for the IMX708 full sensor
# readout (4608x2592 -> ENOMEM at camera start), so we use the 2x2-binned mode.
# This is still the FULL field of view (no crop), just at lower resolution; the
# ROI is applied later in preprocessing. Matches the gateway's CAMERA_RESOLUTION.
CAPTURE_SIZE = (2304, 1296)

# Higher than the gateway's live-capture quality (90): this is archival training
# data that may be cropped/recompressed downstream, so we keep more detail now.
DATASET_JPEG_QUALITY = 95

# Compact, filesystem-safe UTC stamp for filenames. Microsecond precision so two
# captures in the same second cannot collide on the same path.
FILENAME_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S_%fZ"

# Manifest schema. Kept as a constant so the header and every row stay in sync.
MANIFEST_COLUMNS = ("session_id", "class", "filename", "timestamp", "width", "height")

# Layout, relative to this file: dataset/capture_dataset.py -> dataset/ is parent.
_DATASET_ROOT = Path(__file__).resolve().parent
RAW_ROOT = _DATASET_ROOT / "raw"
MANIFEST_PATH = _DATASET_ROOT / "manifest.csv"

# Characters allowed in a session_id so it maps safely to a directory name (no
# path separators, traversal, or spaces). Validated rather than silently
# rewritten, since the id is also recorded verbatim in the manifest.
_SESSION_ID_ALLOWED = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
)


def _prompt_session_id() -> str:
    """Read a non-empty, filesystem-safe session id, re-prompting on bad input."""
    while True:
        value = input('session_id (e.g. "2026-05-30_morning_cloudy"): ').strip()
        if not value:
            print("  session_id cannot be empty.")
            continue
        if not set(value) <= _SESSION_ID_ALLOWED:
            print("  use only letters, digits, '-', '_' and '.' (no spaces/slashes).")
            continue
        return value


def _prompt_class() -> str:
    """Read one of VALID_CLASSES, re-prompting on bad input."""
    options = " / ".join(VALID_CLASSES)
    while True:
        value = input(f"class ({options}): ").strip().lower()
        if value in VALID_CLASSES:
            return value
        print(f"  invalid class; choose one of: {options}")


def _ensure_writable_dir(path: Path) -> None:
    """Create the directory (with parents) and verify it is writable.

    Surfaces a clear, actionable error instead of failing deep inside the
    capture loop when the target is not writable (e.g. wrong owner, read-only
    mount).
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SystemExit(f"cannot create output directory {path}: {exc}") from exc
    if not (path.is_dir() and os.access(path, os.W_OK)):
        raise SystemExit(f"output directory is not writable: {path}")


@contextmanager
def _open_camera() -> Iterator[Picamera2]:
    """Start the camera at full sensor resolution; always stop it on exit.

    The finally block guarantees a clean release even on KeyboardInterrupt or an
    exception during capture, so the sensor is never left running.
    """
    try:
        camera = Picamera2()
    except Exception as exc:  # camera missing, busy, or driver error
        raise SystemExit(f"camera unavailable: {exc}") from exc

    try:
        still_config = camera.create_still_configuration(
            main={"size": CAPTURE_SIZE, "format": CAMERA_PIXEL_FORMAT}
        )
        camera.configure(still_config)
        camera.start()
    except Exception as exc:
        camera.close()
        raise SystemExit(f"camera start failed: {exc}") from exc

    print(f"camera started at {CAPTURE_SIZE[0]}x{CAPTURE_SIZE[1]} (full frame)")
    try:
        yield camera
    finally:
        try:
            camera.stop()
        finally:
            camera.close()
            print("camera stopped.")


def _capture_bgr(camera: Picamera2):
    """Capture one full frame and return it BGR-ordered for OpenCV."""
    frame = camera.capture_array()
    if frame is None or frame.size == 0:
        raise RuntimeError("capture returned an empty frame")
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _append_manifest_row(row: dict[str, object]) -> None:
    """Append one row to the manifest, writing the header on first creation."""
    write_header = not MANIFEST_PATH.exists() or MANIFEST_PATH.stat().st_size == 0
    with MANIFEST_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _count_existing_jpgs(directory: Path) -> int:
    """How many .jpg files the class folder already holds (across past runs)."""
    return sum(1 for _ in directory.glob("*.jpg"))


def _run_session(camera: Picamera2, session_id: str, image_class: str) -> None:
    """Interactive capture loop for one (session, class) pair."""
    out_dir = RAW_ROOT / session_id / image_class
    _ensure_writable_dir(out_dir)

    count = _count_existing_jpgs(out_dir)
    print(
        f"\nsession '{session_id}' / class '{image_class}': "
        f"{count} image(s) already present."
    )
    print("Press ENTER to capture, or 'q' then ENTER to quit.\n")

    while True:
        command = input("[ENTER]=capture  q=quit > ").strip().lower()
        if command == "q":
            break
        if command != "":
            print("  unrecognised input; press ENTER to capture or 'q' to quit.")
            continue

        captured_at = datetime.now(timezone.utc)
        try:
            frame_bgr = _capture_bgr(camera)
        except Exception as exc:  # one bad frame must not end the session
            print(f"  capture failed, not saved: {exc}")
            continue

        filename = f"{image_class}_{captured_at.strftime(FILENAME_TIMESTAMP_FORMAT)}.jpg"
        height, width = frame_bgr.shape[0], frame_bgr.shape[1]
        if not cv2.imwrite(
            str(out_dir / filename),
            frame_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), DATASET_JPEG_QUALITY],
        ):
            print(f"  failed to write image: {out_dir / filename}")
            continue

        _append_manifest_row(
            {
                "session_id": session_id,
                "class": image_class,
                "filename": filename,
                "timestamp": captured_at.isoformat(),
                "width": width,
                "height": height,
            }
        )
        count += 1
        print(f"  saved {filename}  ->  {image_class}: {count} image(s) this class.")


def main() -> None:
    session_id = _prompt_session_id()
    image_class = _prompt_class()
    try:
        with _open_camera() as camera:
            _run_session(camera, session_id, image_class)
    except KeyboardInterrupt:
        # Clean exit: _open_camera's finally has already released the sensor.
        print("\ninterrupted; session ended.")
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
