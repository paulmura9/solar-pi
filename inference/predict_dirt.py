"""Standalone dirt-detection inference: one image in, class + confidence out.

NOT part of the production gateway. This script does not touch MQTT, the
WebSocket, Supabase, or any pi_gateway module. It exists to sanity-check a
trained TFLite model by hand:

    image path -> preprocess -> TFLite model -> predicted class to stdout.

Runtime is ai-edge-litert (the maintained successor to the deprecated
tflite-runtime). Image I/O uses OpenCV, already a repo dependency.

CRITICAL: preprocessing MUST match training exactly. This script reuses the
production transform - pi_gateway.preprocessing.prepare_for_tflite (perspective
warp + resize + normalize) - so it stays aligned with the v4 model and the
gateway. Class ordering still has to match training (see CLASS_LABELS).

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

# Reuse the production preprocessing so this script stays aligned with the v4
# model and the gateway. Make the sibling pi_gateway package importable when run
# as a standalone script (python inference/predict_dirt.py) by adding the repo
# root to sys.path before importing it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi_gateway.preprocessing import EXPECTED_INPUT_SHAPE, prepare_for_tflite

# --- Training-coupled constants (CONFIRM FROM COLAB) --------------------------

# Class labels in the EXACT order the model's output neurons were trained on.
# The argmax of the output vector indexes into this tuple, so a wrong order
# silently mislabels every prediction. CONFIRM FROM COLAB that index 0 == clean,
# 1 == slightly_dirty, 2 == dirty (matches dataset/capture_dataset.VALID_CLASSES).
CLASS_LABELS = ("clean", "slightly_dirty", "dirty")

# Whether the exported model's final layer emits raw logits (True) or already
# applies softmax (False). Determines whether we softmax the output to obtain
# probabilities; double-softmaxing distorts the reported confidence even though
# the argmax is unchanged. CONFIRM FROM COLAB the model's output activation.
MODEL_OUTPUT_IS_LOGITS = False

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
    model_input = prepare_for_tflite(image_bgr)
    raw_output, elapsed_ms = _infer(interpreter, model_input)
    _report(raw_output, elapsed_ms)


if __name__ == "__main__":
    sys.exit(main())
