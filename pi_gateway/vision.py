"""Edge dirt-detection inference (runs strictly on the Pi).

Loads the trained TFLite model ONCE (ai-edge-litert, the maintained successor to
tflite-runtime) and exposes a pure inference call that turns an in-memory BGR
frame into a structured VisionResult. No disk I/O, no DB writes: the Pi uploads
images to Storage and forwards results over the WebSocket; Express persists the
vision_results row.

PREPROCESSING - CRITICAL: must be byte-for-byte equivalent to training. The v4
model was trained on images straightened by a fixed perspective warp, so inference
applies the identical warp before resizing. That transform lives in
pi_gateway/preprocessing.py (warp_panel / prepare_for_tflite) and is shared with
the surface-analysis pipeline so both work on the same straightened panel. Any
divergence invalidates every prediction.

Class order is fixed by training: index 0=clean, 1=slightly_dirty, 2=dirty. The
exported model already applies softmax (Dense(3, activation='softmax')), so the
output tensor is used directly as probabilities and softmax is NOT re-applied.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import cv2
import numpy as np

# ai-edge-litert is the maintained replacement for the deprecated tflite-runtime.
from ai_edge_litert.interpreter import Interpreter

from . import config, protocol
from .logging_utils import log
from .preprocessing import (
    EXPECTED_INPUT_SHAPE,
    INPUT_CHANNELS,
    prepare_for_tflite,
    warp_panel,
)
from .storage import StorageClient
from .surface_analysis import build_surface_overlay

WsSender = Callable[[dict[str, Any]], Awaitable[None]]

# --- Class vocabulary (order fixed by training; do not reorder) ---------------
CLASS_CLEAN = "clean"
CLASS_SLIGHTLY_DIRTY = "slightly_dirty"
CLASS_DIRTY = "dirty"
CLASS_LABELS = (CLASS_CLEAN, CLASS_SLIGHTLY_DIRTY, CLASS_DIRTY)
# Indices derived from the label order so the weighting below cannot drift out of
# sync with CLASS_LABELS if the tuple is ever edited.
INDEX_SLIGHTLY_DIRTY = CLASS_LABELS.index(CLASS_SLIGHTLY_DIRTY)
INDEX_DIRTY = CLASS_LABELS.index(CLASS_DIRTY)

# Model input contract (EXPECTED_INPUT_SHAPE, INPUT_CHANNELS) lives in
# preprocessing.py alongside the transform that produces it.

# --- Derived 0..100 metrics ---------------------------------------------------
# Collapse the 3-class probability distribution into a single dirt score: a
# slightly_dirty panel counts as half-dirty, a dirty panel as fully dirty, clean
# contributes 0. So dirt_level = P(slightly_dirty)*50 + P(dirty)*100.
SLIGHTLY_DIRTY_WEIGHT_PERCENT = 50
DIRTY_WEIGHT_PERCENT = 100
# Cleanliness is the complement of the dirt level on the same 0..100 scale.
FULL_PERCENT = 100
# Percent metrics are reported rounded to 2 decimals.
PERCENT_DECIMALS = 2

# --- Class thresholds on the 0..100 dirt scale --------------------------------
# Reported class, cleaning flag and dirt_level_percent must stay mutually
# consistent, so the class is derived from the continuous dirt score rather than
# the raw argmax (which can disagree with the weighted score). Bands:
# clean <= 33 < slightly_dirty <= 66 < dirty.
DIRT_CLEAN_MAX_PERCENT = 33
DIRT_SLIGHTLY_MAX_PERCENT = 66
# Cleaning is required once the panel reaches the dirty band.
CLEANING_REQUIRED_PERCENT = 66

# --- Pre-inference quality gate ----------------------------------------------
QUALITY_REASON_TOO_DARK = "too_dark"
QUALITY_REASON_TOO_BRIGHT = "too_bright"
QUALITY_REASON_NOT_PANEL = "not_panel"
QUALITY_REASON_LOW_DETAIL = "low_detail"


def crop_to_roi(frame_bgr: np.ndarray) -> np.ndarray:
    """Crop to the fixed obstruction-check ROI (config.DIRT_ROI_*); full-frame fallback.

    Used only by the pre-inference quality gate to sample the panel region for
    obstruction. The model and surface-analysis preprocessing use the perspective
    warp (preprocessing.py), not this crop. If the ROI does not fit the frame, log
    a warning and return the full frame (a suboptimal check beats a crash).
    """
    height, width = frame_bgr.shape[0], frame_bgr.shape[1]
    x, y, w, h = config.DIRT_ROI_X, config.DIRT_ROI_Y, config.DIRT_ROI_W, config.DIRT_ROI_H
    if x < 0 or y < 0 or x + w > width or y + h > height:
        log("warning", "vision_roi_out_of_bounds", roi=[x, y, w, h], frame=[width, height])
        return frame_bgr
    return frame_bgr[y : y + h, x : x + w]


def check_frame_quality(frame_bgr: np.ndarray) -> tuple[bool, Optional[str]]:
    """Quality gate run on the ROI crop before inference.

    Rejects obstructed frames so the model is never fed garbage. Checks, in order
    (first match wins): too dark (low grayscale mean), too bright/overexposed
    (high grayscale mean), not the panel (red dominates blue - skin/an object over
    the dark-blue panel), too little detail (low grayscale std, e.g. a uniform
    surface covering the lens). Returns (True, None) when usable, else
    (False, reason). Thresholds are heuristics in config (DIRT_QUALITY_*).
    """
    crop = crop_to_roi(frame_bgr)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    mean = float(gray.mean())
    if mean < config.DIRT_QUALITY_MIN_MEAN:
        return False, QUALITY_REASON_TOO_DARK
    if mean > config.DIRT_QUALITY_MAX_MEAN:
        return False, QUALITY_REASON_TOO_BRIGHT

    mean_b, _mean_g, mean_r = (float(channel) for channel in cv2.mean(crop)[:3])
    if mean_r - mean_b > config.DIRT_QUALITY_RED_DOMINANCE:
        return False, QUALITY_REASON_NOT_PANEL

    if float(gray.std()) < config.DIRT_QUALITY_MIN_STD:
        return False, QUALITY_REASON_LOW_DETAIL
    return True, None


class VisionError(RuntimeError):
    """Raised when the model cannot be loaded or an inference fails."""


@dataclass(frozen=True)
class VisionResult:
    """Structured outcome of one dirt-detection inference."""

    predicted_class: str
    probabilities: dict[str, float]  # label -> probability (sums to ~1.0)
    confidence: float  # max softmax probability, 0..1
    dirt_level_percent: float
    cleanliness_percent: float
    cleaning_required: bool


class DirtDetector:
    """Owns the TFLite interpreter; loads the model once and runs inference.

    The interpreter is loaded a single time at construction. invoke() is not
    reentrant, so a lock serializes inferences: the periodic vision loop and the
    manual-capture handler may both call predict() concurrently.
    """

    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise VisionError(f"dirt model not found: {path}")

        try:
            interpreter = Interpreter(model_path=str(path))
            interpreter.allocate_tensors()
        except Exception as exc:  # corrupt / incompatible model file
            raise VisionError(f"failed to load dirt model {path}: {exc}") from exc

        input_detail = interpreter.get_input_details()[0]
        actual_shape = tuple(int(dim) for dim in input_detail["shape"])
        if actual_shape != EXPECTED_INPUT_SHAPE:
            raise VisionError(
                f"model input shape {actual_shape} != expected {EXPECTED_INPUT_SHAPE}"
            )
        if input_detail["dtype"] != np.float32:
            raise VisionError(
                f"model expects {input_detail['dtype']} input; preprocessing produces float32"
            )

        self._interpreter = interpreter
        self._input_index = input_detail["index"]
        self._output_index = interpreter.get_output_details()[0]["index"]
        self._lock = threading.Lock()
        log("info", "vision_model_loaded", model_path=str(path))

    @classmethod
    def from_config(cls) -> "DirtDetector":
        return cls(config.DIRT_MODEL_PATH)

    def predict(self, frame_bgr: np.ndarray) -> VisionResult:
        """Run inference on one in-memory BGR frame. Blocking; call off the loop.

        Pure with respect to the model (no I/O). Preprocessing (perspective warp +
        resize + normalize) is the shared prepare_for_tflite. Raises VisionError on
        an invalid frame or an unexpected model output so the caller can log and
        continue.
        """
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            raise VisionError("empty frame")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != INPUT_CHANNELS:
            raise VisionError(f"expected an HxWx{INPUT_CHANNELS} BGR frame, got shape {frame_bgr.shape}")

        model_input = prepare_for_tflite(frame_bgr)
        with self._lock:
            self._interpreter.set_tensor(self._input_index, model_input)
            self._interpreter.invoke()
            raw_output = self._interpreter.get_tensor(self._output_index)[0]

        if raw_output.shape != (len(CLASS_LABELS),):
            raise VisionError(
                f"model output {raw_output.shape} != one value per class {CLASS_LABELS}"
            )

        # The model already applies softmax, so the output is used directly.
        probabilities = [float(value) for value in raw_output]
        # argmax is the model's most-likely class; kept only to report confidence
        # (its softmax probability). The reported class is NOT taken from here - it
        # is derived from dirt_level_percent below so it cannot contradict the score.
        predicted_index = int(np.argmax(raw_output))

        dirt_level_percent = round(
            probabilities[INDEX_SLIGHTLY_DIRTY] * SLIGHTLY_DIRTY_WEIGHT_PERCENT
            + probabilities[INDEX_DIRTY] * DIRTY_WEIGHT_PERCENT,
            PERCENT_DECIMALS,
        )
        cleanliness_percent = round(FULL_PERCENT - dirt_level_percent, PERCENT_DECIMALS)

        # Class and cleaning flag both derive from the continuous dirt score, so all
        # three reported outputs are consistent functions of dirt_level_percent.
        if dirt_level_percent <= DIRT_CLEAN_MAX_PERCENT:
            predicted_class = CLASS_CLEAN
        elif dirt_level_percent <= DIRT_SLIGHTLY_MAX_PERCENT:
            predicted_class = CLASS_SLIGHTLY_DIRTY
        else:
            predicted_class = CLASS_DIRTY

        return VisionResult(
            predicted_class=predicted_class,
            probabilities=dict(zip(CLASS_LABELS, probabilities)),
            confidence=probabilities[predicted_index],
            dirt_level_percent=dirt_level_percent,
            cleanliness_percent=cleanliness_percent,
            cleaning_required=dirt_level_percent >= CLEANING_REQUIRED_PERCENT,
        )


def _build_and_upload_overlay(
    frame_bgr: np.ndarray, image_path: str, storage: StorageClient
) -> Optional[str]:
    """Best-effort: build the classical-CV surface overlay and upload it.

    Returns the Storage object path, or None when generation/upload fails (the
    vision_result is still sent, just without a processed image). The overlay is
    stored under the surface/ prefix reusing the original frame's object name, so
    each overlay correlates with its source image. Blocking; call in a worker
    thread.
    """
    try:
        # Same straightened panel the model sees. warp_panel returns RGB; convert
        # back to BGR for the BGR-oriented overlay (correct red highlight + JPEG).
        warped_bgr = cv2.cvtColor(warp_panel(frame_bgr), cv2.COLOR_RGB2BGR)
        overlay_jpeg = build_surface_overlay(warped_bgr)
        object_name = image_path.split("/", 1)[1] if "/" in image_path else image_path
        processed_path = f"{config.STORAGE_SURFACE_PREFIX}/{object_name}"
        storage.upload_jpeg(processed_path, overlay_jpeg)
        return processed_path
    except Exception as exc:
        log("warning", "surface_overlay_failed", image_path=image_path, error=str(exc))
        return None


async def detect_and_report(
    detector: DirtDetector,
    frame_bgr: np.ndarray,
    image_path: str,
    captured_at: str,
    storage: StorageClient,
    ws_send: WsSender,
) -> Optional[VisionResult]:
    """Run inference off the event loop and report a vision_result over the WS.

    Shared by the periodic vision loop and the manual-capture handler so the
    inference + reporting path exists once. A pre-inference quality gate rejects
    obstructed frames: when it fails, the model is NOT run - a neutral
    vision_result is sent with quality_ok=false and the cause, and None is
    returned. Otherwise it runs DirtDetector.predict, builds the best-effort
    surface overlay (classical CV, auxiliary - see surface_analysis), and sends a
    quality_ok=true result. Raises on inference failure so the caller can log and
    continue; a failed overlay only leaves processed_image_path as None.
    """
    quality_ok, quality_reason = await asyncio.to_thread(check_frame_quality, frame_bgr)
    if not quality_ok:
        log("warning", "vision_quality_gate_blocked", image_path=image_path, reason=quality_reason)
        # Send the captured frame (already uploaded, useful for debug) with neutral
        # class/percentages so the obstruction is visible without a model run.
        await ws_send(
            protocol.build_vision_result(
                predicted_class=None,
                dirt_level_percent=0.0,
                cleanliness_percent=0.0,
                cleaning_required=False,
                confidence=0.0,
                image_path=image_path,
                processed_image_path=None,
                captured_at=captured_at,
                quality_ok=False,
                quality_reason=quality_reason,
            )
        )
        return None

    result = await asyncio.to_thread(detector.predict, frame_bgr)
    processed_image_path = await asyncio.to_thread(
        _build_and_upload_overlay, frame_bgr, image_path, storage
    )
    await ws_send(
        protocol.build_vision_result(
            predicted_class=result.predicted_class,
            dirt_level_percent=result.dirt_level_percent,
            cleanliness_percent=result.cleanliness_percent,
            cleaning_required=result.cleaning_required,
            confidence=result.confidence,
            image_path=image_path,
            processed_image_path=processed_image_path,
            captured_at=captured_at,
            quality_ok=True,
            quality_reason=None,
        )
    )
    return result
