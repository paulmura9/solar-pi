"""Edge dirt-detection inference (runs strictly on the Pi).

Loads the trained TFLite model ONCE (ai-edge-litert, the maintained successor to
tflite-runtime) and exposes a pure inference call that turns an in-memory BGR
frame into a structured VisionResult. No disk I/O, no DB writes: the Pi uploads
images to Storage and forwards results over the WebSocket; Express persists the
vision_results row.

PREPROCESSING - CRITICAL: must be byte-for-byte equivalent to training. The model
was trained on the FULL frame (NO crop / NO ROI): resize to 224x224, BGR->RGB,
normalize /255.0 as float32, add the batch dimension. Any divergence invalidates
every prediction. CameraManager.capture_full_frame() already yields a full-frame
BGR array, which is exactly what feeds in here.

Class order is fixed by training: index 0=clean, 1=slightly_dirty, 2=dirty. The
exported model already applies softmax (Dense(3, activation='softmax')), so the
output tensor is used directly as probabilities and softmax is NOT re-applied.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import cv2
import numpy as np

# ai-edge-litert is the maintained replacement for the deprecated tflite-runtime.
from ai_edge_litert.interpreter import Interpreter

from . import config, protocol
from .logging_utils import log

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

# --- Model input contract (must match the training preprocessing) ------------
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
INPUT_CHANNELS = 3
PIXEL_MAX_VALUE = 255.0  # 8-bit images -> /255.0 maps [0,255] to [0.0,1.0]
EXPECTED_INPUT_SHAPE = (1, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)

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

    @staticmethod
    def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
        """Turn a full-frame BGR array into the model's (1,224,224,3) float32 input.

        Mirrors training exactly: NO crop, resize to 224x224, BGR->RGB, /255.0
        float32, add the batch axis.
        """
        resized = cv2.resize(frame_bgr, (INPUT_WIDTH, INPUT_HEIGHT))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / PIXEL_MAX_VALUE
        return np.expand_dims(normalized, axis=0)

    def predict(self, frame_bgr: np.ndarray) -> VisionResult:
        """Run inference on one in-memory BGR frame. Blocking; call off the loop.

        Pure with respect to the model (no I/O). Raises VisionError on an invalid
        frame or an unexpected model output so the caller can log and continue.
        """
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            raise VisionError("empty frame")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != INPUT_CHANNELS:
            raise VisionError(f"expected an HxWx{INPUT_CHANNELS} BGR frame, got shape {frame_bgr.shape}")

        model_input = self._preprocess(frame_bgr)
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
        predicted_index = int(np.argmax(raw_output))
        predicted_class = CLASS_LABELS[predicted_index]

        dirt_level_percent = round(
            probabilities[INDEX_SLIGHTLY_DIRTY] * SLIGHTLY_DIRTY_WEIGHT_PERCENT
            + probabilities[INDEX_DIRTY] * DIRTY_WEIGHT_PERCENT,
            PERCENT_DECIMALS,
        )
        cleanliness_percent = round(FULL_PERCENT - dirt_level_percent, PERCENT_DECIMALS)

        return VisionResult(
            predicted_class=predicted_class,
            probabilities=dict(zip(CLASS_LABELS, probabilities)),
            confidence=probabilities[predicted_index],
            dirt_level_percent=dirt_level_percent,
            cleanliness_percent=cleanliness_percent,
            cleaning_required=predicted_class == CLASS_DIRTY,
        )


async def detect_and_report(
    detector: DirtDetector,
    frame_bgr: np.ndarray,
    image_path: str,
    captured_at: str,
    ws_send: WsSender,
) -> VisionResult:
    """Run inference off the event loop and report a vision_result over the WS.

    Shared by the periodic vision loop and the manual-capture handler so the
    inference + reporting path exists once. The pure inference is
    DirtDetector.predict; this wrapper adds the blocking-call offload and the WS
    envelope. Raises on inference failure so the caller can log and continue.
    processed_image_path is None until overlay generation exists.
    """
    result = await asyncio.to_thread(detector.predict, frame_bgr)
    await ws_send(
        protocol.build_vision_result(
            dirt_level_percent=result.dirt_level_percent,
            cleanliness_percent=result.cleanliness_percent,
            cleaning_required=result.cleaning_required,
            confidence=result.confidence,
            image_path=image_path,
            processed_image_path=None,
            captured_at=captured_at,
        )
    )
    return result
