"""Centralized image preprocessing for the dirt-detection pipelines.

Single source of truth for the perspective warp that straightens the side-on view
of the solar panel, plus the TFLite input transform. Both the ML inference
(vision.DirtDetector) and the classical surface analysis (surface_analysis) run on
the SAME warped image so the dirt heatmap is aligned with what the model sees.

This must stay byte-for-byte identical to the Colab training preprocessing:
BGR -> RGB, warpPerspective with M, then (TFLite only) resize to 224x224 and /255.
No CLAHE, no other filters.

PANEL_PTS depends on the PHYSICAL camera position used to build the dataset.
Recalibrate PANEL_PTS (and thus M) if the camera is moved.
"""
from __future__ import annotations

import cv2
import numpy as np

PANEL_PTS = np.float32([[550, 35], [2140, 140], [2300, 1115], [535, 1290]])

WARP_W, WARP_H = 600, 400
DST_PTS = np.float32([[0, 0], [WARP_W, 0], [WARP_W, WARP_H], [0, WARP_H]])
M = cv2.getPerspectiveTransform(PANEL_PTS, DST_PTS)

TFLITE_INPUT_SIZE = 224
INPUT_CHANNELS = 3
PIXEL_MAX_VALUE = 255.0
EXPECTED_INPUT_SHAPE = (1, TFLITE_INPUT_SIZE, TFLITE_INPUT_SIZE, INPUT_CHANNELS)


def warp_panel(img_bgr: np.ndarray) -> np.ndarray:
    """Straighten the panel: BGR -> RGB, then perspective warp to WARP_W x WARP_H.

    Returns an RGB image (the channel order the model was trained on).
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.warpPerspective(rgb, M, (WARP_W, WARP_H))


def prepare_for_tflite(img_bgr: np.ndarray) -> np.ndarray:
    """Turn a camera BGR frame into the model input: (1, 224, 224, 3) float32.

    warp_panel (BGR->RGB + perspective warp), then resize to the network input
    size and normalize to [0, 1]. Identical to the Colab training transform.
    """
    warped = warp_panel(img_bgr)
    resized = cv2.resize(warped, (TFLITE_INPUT_SIZE, TFLITE_INPUT_SIZE))
    normalized = resized.astype(np.float32) / PIXEL_MAX_VALUE
    return np.expand_dims(normalized, axis=0)
