"""Standalone dirt-detection inference: one image in, class + confidence out.

NOT part of the production gateway. This script does not touch MQTT, the
WebSocket, Supabase, or any pi_gateway module. It exists to sanity-check a
trained TFLite model by hand:

    image path -> preprocess -> TFLite model -> predicted class to stdout.

Runtime is ai-edge-litert (the maintained successor to the deprecated
tflite-runtime). Image I/O uses OpenCV, already a repo dependency.

CRITICAL: the preprocessing below MUST be byte-for-byte equivalent to the
preprocessing used during training in Colab. Any divergence (ROI, resize
interpolation, channel order, normalization, or class ordering) makes the
predictions meaningless even when the script runs without error. The values
marked "CONFIRM FROM COLAB" are the training-coupled knobs to verify before
trusting any output.

Usage:
    python inference/predict_dirt.py path/to/image.jpg
    python inference/predict_dirt.py path/to/image.jpg --model models/dirt_detection.tflite
    DIRT_MODEL_PATH=models/dirt_detection.tflite python inference/predict_dirt.py path/to/image.jpg
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# ai-edge-litert is the maintained replacement for the deprecated tflite-runtime.
from ai_edge_litert.interpreter import Interpreter

# --- Training-coupled constants (CONFIRM FROM COLAB) --------------------------

# Class labels in the EXACT order the model's output neurons were trained on.
# The argmax of the output vector indexes into this tuple, so a wrong order
# silently mislabels every prediction. CONFIRM FROM COLAB that index 0 == clean,
# 1 == slightly_dirty, 2 == dirty (matches dataset/capture_dataset.VALID_CLASSES).
CLASS_LABELS = ("clean", "slightly_dirty", "dirty")

# Region of interest cropped from the full frame before resizing, as
# (x, y, width, height) in pixels of the captured image. The dataset is stored
# full-frame (see dataset/capture_dataset.py), and the ROI is applied here in
# preprocessing exactly as in training.
# PLACEHOLDER: replace with the real coordinates established in Colab. These must
# be identical to the crop used to build the training set. CONFIRM FROM COLAB.
ROI_X = 0
ROI_Y = 0
ROI_WIDTH = 2304   # full-frame width  from CAPTURE_SIZE; narrow once ROI is fixed
ROI_HEIGHT = 1296  # full-frame height from CAPTURE_SIZE; narrow once ROI is fixed
ROI = (ROI_X, ROI_Y, ROI_WIDTH, ROI_HEIGHT)

# Whether to reorder channels BGR -> RGB before feeding the model. OpenCV decodes
# images as BGR; if the Colab pipeline read images as RGB (e.g. via PIL or
# tf.io.decode_image / Keras), the model expects RGB and this MUST be True.
# CONFIRM FROM COLAB which channel order training used.
CONVERT_BGR_TO_RGB = True

# Whether the exported model's final layer emits raw logits (True) or already
# applies softmax (False). Determines whether we softmax the output to obtain
# probabilities; double-softmaxing distorts the reported confidence even though
# the argmax is unchanged. CONFIRM FROM COLAB the model's output activation.
MODEL_OUTPUT_IS_LOGITS = False

# --- Fixed model-contract constants ------------------------------------------

# Network input is 224x224 RGB/BGR, float32, scaled to [0, 1].
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
INPUT_CHANNELS = 3
PIXEL_MAX_VALUE = 255.0  # 8-bit images -> /255.0 maps [0,255] to [0.0,1.0]
BATCH_DIM = 1
EXPECTED_INPUT_SHAPE = (BATCH_DIM, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)

# Default model location, overridable by --model or the DIRT_MODEL_PATH env var.
MODEL_PATH_ENV_VAR = "DIRT_MODEL_PATH"
DEFAULT_MODEL_PATH = "models/dirt_detection.tflite"

MS_PER_SECOND = 1000.0


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments: required image path, optional model path override."""
    parser = argparse.ArgumentParser(
        description="Run the dirt-detection TFLite model on a single image."
    )
    parser.add_argument("image", help="path to the input image (e.g. dataset/raw/.../x.jpg)")
    parser.add_argument(
        "--model",
        default=os.environ.get(MODEL_PATH_ENV_VAR, DEFAULT_MODEL_PATH),
        help=(
            "path to the .tflite model "
            f"(default: ${MODEL_PATH_ENV_VAR} or {DEFAULT_MODEL_PATH})"
        ),
    )
    return parser.parse_args()


def _load_interpreter(model_path: Path) -> Interpreter:
    """Load the TFLite model and validate its input contract.

    Fails with a clear message when the model file is missing or when its input
    tensor does not match the (1, 224, 224, 3) float32 layout this preprocessing
    produces, instead of feeding it a mismatched tensor and getting garbage.
    """
    if not model_path.is_file():
        raise SystemExit(
            f"model not found: {model_path}\n"
            f"set --model or ${MODEL_PATH_ENV_VAR}, or place the trained model there."
        )

    try:
        interpreter = Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
    except Exception as exc:  # corrupt/incompatible model file
        raise SystemExit(f"failed to load TFLite model {model_path}: {exc}") from exc

    input_detail = interpreter.get_input_details()[0]
    actual_shape = tuple(int(dim) for dim in input_detail["shape"])
    if actual_shape != EXPECTED_INPUT_SHAPE:
        raise SystemExit(
            f"model input shape {actual_shape} != expected {EXPECTED_INPUT_SHAPE}; "
            "the preprocessing in this script does not match this model."
        )
    if input_detail["dtype"] != np.float32:
        raise SystemExit(
            f"model expects {input_detail['dtype']} input, but this script produces "
            "float32 [0,1]. A quantized model needs different preprocessing."
        )
    return interpreter


def _load_image(image_path: Path) -> np.ndarray:
    """Read an image from disk as a BGR uint8 array, or fail clearly.

    cv2.imread returns None for a missing, unreadable, or corrupt file without
    raising, so the None case is checked explicitly.
    """
    if not image_path.is_file():
        raise SystemExit(f"image not found: {image_path}")
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise SystemExit(f"could not decode image (missing or corrupt): {image_path}")
    return image_bgr


def _preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """Turn a BGR image into the model's (1, 224, 224, 3) float32 input.

    MUST mirror the training preprocessing exactly (see module docstring):
      1. crop to ROI,
      2. resize to 224x224,
      3. (optional) BGR -> RGB to match the training channel order,
      4. normalize to [0, 1] as float32,
      5. add the batch dimension.
    """
    height, width = image_bgr.shape[0], image_bgr.shape[1]
    x, y, w, h = ROI
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
        raise SystemExit(
            f"ROI {ROI} does not fit within image of size {width}x{height}; "
            "check the ROI constants against the captured frame size."
        )

    # 1. Crop to the same region the model was trained on.
    cropped = image_bgr[y : y + h, x : x + w]

    # 2. Resize to the network input size.
    resized = cv2.resize(cropped, (INPUT_WIDTH, INPUT_HEIGHT))

    # 3. Match the training channel order.
    if CONVERT_BGR_TO_RGB:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 4. Normalize to [0, 1] float32.
    normalized = resized.astype(np.float32) / PIXEL_MAX_VALUE

    # 5. Add the leading batch dimension (axis 0) -> (1, 224, 224, 3).
    return np.expand_dims(normalized, axis=0)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D vector of scores."""
    shifted = logits - np.max(logits)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated)


def _infer(interpreter: Interpreter, model_input: np.ndarray) -> Tuple[np.ndarray, float]:
    """Run one inference, returning the output vector and elapsed time in ms."""
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, model_input)
    started = time.perf_counter()
    interpreter.invoke()
    elapsed_ms = (time.perf_counter() - started) * MS_PER_SECOND

    raw_output = interpreter.get_tensor(output_index)[0]  # drop the batch dimension
    if raw_output.shape != (len(CLASS_LABELS),):
        raise SystemExit(
            f"model output has {raw_output.shape} values, expected {len(CLASS_LABELS)} "
            f"(one per class in {CLASS_LABELS}); class list and model disagree."
        )
    return raw_output, elapsed_ms


def _report(raw_output: np.ndarray, elapsed_ms: float) -> None:
    """Print the predicted class, confidence, per-class probabilities, and timing."""
    probabilities = _softmax(raw_output) if MODEL_OUTPUT_IS_LOGITS else raw_output
    predicted_index = int(np.argmax(probabilities))

    print(f"predicted class: {CLASS_LABELS[predicted_index]}")
    print(f"confidence:      {probabilities[predicted_index]:.4f}")
    print("probabilities:")
    for label, probability in zip(CLASS_LABELS, probabilities):
        print(f"  {label:<15} {probability:.4f}")
    print(f"inference time:  {elapsed_ms:.2f} ms")


def main() -> None:
    args = _parse_args()
    interpreter = _load_interpreter(Path(args.model))
    image_bgr = _load_image(Path(args.image))
    model_input = _preprocess(image_bgr)
    raw_output, elapsed_ms = _infer(interpreter, model_input)
    _report(raw_output, elapsed_ms)


if __name__ == "__main__":
    sys.exit(main())
