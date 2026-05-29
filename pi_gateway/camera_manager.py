"""CameraManager: the single owner of the Picamera2 instance.

Every capture is serialized through a threading.Lock so the manual CAPTURE_IMAGE
handler and any future periodic ML loop share one camera without concurrent
access (Picamera2 does not allow concurrent captures). Camera-unavailable and
capture-failure conditions are surfaced explicitly as CameraError.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np
from picamera2 import Picamera2

from . import config
from .logging_utils import log


class CameraError(RuntimeError):
    """Raised when the camera cannot be started or a capture fails."""


class CameraManager:
    def __init__(
        self,
        resolution: Tuple[int, int] = config.CAMERA_RESOLUTION,
        pixel_format: str = config.CAMERA_PIXEL_FORMAT,
    ) -> None:
        self._resolution = resolution
        self._pixel_format = pixel_format
        self._lock = threading.Lock()
        self._camera: Optional[Picamera2] = None

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Acquire the camera lock with a bounded wait.

        A wedged capture must never block other captures (or the future periodic
        ML loop) forever, so acquisition times out and raises CameraError instead
        of waiting indefinitely.
        """
        acquired = self._lock.acquire(timeout=config.CAMERA_LOCK_TIMEOUT_S)
        if not acquired:
            raise CameraError("camera busy: lock not acquired within timeout")
        try:
            yield
        finally:
            self._lock.release()

    def start(self) -> None:
        """Initialize and start the sensor. Idempotent and lock-guarded.

        Safe to call again after a previous failure: the camera is only retained
        once it has started cleanly, so a later call will retry initialization.
        """
        with self._locked():
            if self._camera is not None:
                return
            try:
                camera = Picamera2()
                still_config = camera.create_still_configuration(
                    main={"size": self._resolution, "format": self._pixel_format}
                )
                camera.configure(still_config)
                camera.start()
            except Exception as exc:
                raise CameraError(f"camera start failed: {exc}") from exc
            self._camera = camera
            log("info", "camera_started", resolution=list(self._resolution))

    def stop(self) -> None:
        """Stop and release the sensor. Idempotent and lock-guarded."""
        with self._locked():
            if self._camera is None:
                return
            try:
                self._camera.stop()
                self._camera.close()
            finally:
                self._camera = None
                log("info", "camera_stopped")

    def capture_full_frame(self) -> Tuple[np.ndarray, int, int]:
        """Capture one full frame (no crop). Returns (bgr_frame, width, height).

        Lock-guarded so captures never overlap. The returned array is BGR-ordered
        and ready for cv2 encoding (picamera2 yields RGB order on the Pi; see
        scripts/camera_test.py). The color conversion runs outside the lock to
        keep the critical section as short as the sensor read.
        """
        with self._locked():
            if self._camera is None:
                raise CameraError("camera not started")
            try:
                frame = self._camera.capture_array()
            except Exception as exc:
                raise CameraError(f"capture failed: {exc}") from exc

        if frame is None or frame.size == 0:
            raise CameraError("capture returned an empty frame")

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        height, width = frame_bgr.shape[0], frame_bgr.shape[1]
        return frame_bgr, width, height
