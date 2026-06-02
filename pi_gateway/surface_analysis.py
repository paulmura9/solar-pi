"""Classical-OpenCV surface-deposit overlay (independent of the ML model).

Estimates likely surface deposits on the panel by pure image processing - local
regions brighter than the surrounding panel once uneven background lighting is
removed - and paints them as a semi-transparent highlight over the straightened
panel image.

IMPORTANT: this is a visual ESTIMATE for the operator, produced by classic image
processing. It is NOT the CNN's decision and does not represent what the
dirt-detection model "sees". It only fills processed_image_path in vision_result;
the predicted class and the dirt/cleanliness percentages come solely from the
model (see vision.py).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import config

# Drop dirt-mask contours whose bounding-box aspect ratio (width/height) exceeds
# this: long thin shapes are the panel's light horizontal bus bars, not dirt.
# Lower values filter lines more aggressively but risk cutting elongated dirt.
ASPECT_RATIO_MAX = 8


class SurfaceAnalysisError(RuntimeError):
    """Raised when the surface overlay cannot be produced."""


def _drop_bus_bar_contours(dirt_mask: np.ndarray) -> np.ndarray:
    """Rebuild the dirt mask without long thin (bus-bar) components.

    Filters the mask's external contours by bounding-box aspect ratio: a contour
    wider than ASPECT_RATIO_MAX:1 is the panel's horizontal bus bar, not dirt, so
    it is dropped; compact blobs are kept and refilled. Shape-based, so dirt lying
    on a bar (a compact blob) survives while the bar itself is removed.
    """
    contours, _ = cv2.findContours(dirt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = np.zeros_like(dirt_mask)
    for contour in contours:
        _x, _y, width, height = cv2.boundingRect(contour)
        if width / height <= ASPECT_RATIO_MAX:
            cv2.drawContours(kept, [contour], -1, 255, thickness=cv2.FILLED)
    return kept


def build_surface_overlay(panel_bgr: np.ndarray) -> bytes:
    """Return JPEG bytes of the straightened panel with deposits highlighted.

    `panel_bgr` must be the straightened panel (preprocessing.warp_panel output,
    converted to BGR) so the overlay aligns with the region the model analyzes.
    Pure image processing (see module docstring): a large Gaussian blur estimates
    the uneven background illumination; a saturating subtract keeps only pixels
    brighter than that background (deposits show as light specks on the darker
    panel); a threshold isolates them; a shape filter drops the long thin bus
    bars (see _drop_bus_bar_contours); and the result is blended as a
    semi-transparent highlight over the original panel image.
    """
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    kernel = (config.SURFACE_BLUR_KERNEL, config.SURFACE_BLUR_KERNEL)
    background = cv2.GaussianBlur(gray, kernel, 0)

    # Saturating subtract: pixels darker than the local background clamp to 0, so
    # only lighter-than-panel regions (candidate deposits) survive.
    diff = cv2.subtract(gray, background)
    _, mask = cv2.threshold(
        diff, config.SURFACE_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY
    )

    # Shape filter (before the overlay, and before any metric derived from the
    # mask) so the panel's light bus bars are not highlighted/counted as dirt.
    mask = _drop_bus_bar_contours(mask)

    overlay = panel_bgr.copy()
    overlay[mask > 0] = config.SURFACE_HIGHLIGHT_COLOR_BGR
    blended = cv2.addWeighted(
        overlay,
        config.SURFACE_OVERLAY_ALPHA,
        panel_bgr,
        1.0 - config.SURFACE_OVERLAY_ALPHA,
        0.0,
    )

    ok, buffer = cv2.imencode(
        ".jpg", blended, [int(cv2.IMWRITE_JPEG_QUALITY), config.CAPTURE_JPEG_QUALITY]
    )
    if not ok:
        raise SurfaceAnalysisError("surface overlay JPEG encoding failed")
    return buffer.tobytes()
