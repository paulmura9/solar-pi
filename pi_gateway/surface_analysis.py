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


class SurfaceAnalysisError(RuntimeError):
    """Raised when the surface overlay cannot be produced."""


def _remove_bus_bars(dirt_mask: np.ndarray) -> np.ndarray:
    """Remove the panel's straight bus bars and cell edges from the dirt mask.

    In the straightened panel the bus bars run strictly horizontal and the cell
    edges strictly vertical, while dirt is isotropic. A morphological opening with a
    long thin horizontal LINE kernel keeps only horizontal runs at least
    config.SURFACE_HLINE_LENGTH px long (the bus bars); the same with a vertical
    kernel of config.SURFACE_VLINE_LENGTH keeps the vertical cell edges. Their union
    is subtracted from the mask. Compact deposits match neither line kernel, so they
    survive even when smaller than the bus bars are thick - the case a uniform
    opening cannot handle, and unlike a bounding-box test it removes bars even where
    they interconnect into one panel-spanning lattice. Dirt lying directly on a bar
    is removed with it (an accepted edge case; off-bar deposits are unaffected).
    """
    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (config.SURFACE_HLINE_LENGTH, 1))
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, config.SURFACE_VLINE_LENGTH))
    h_lines = cv2.morphologyEx(dirt_mask, cv2.MORPH_OPEN, horizontal)
    v_lines = cv2.morphologyEx(dirt_mask, cv2.MORPH_OPEN, vertical)
    lines = cv2.bitwise_or(h_lines, v_lines)
    return cv2.subtract(dirt_mask, lines)


def build_surface_overlay(panel_bgr: np.ndarray) -> bytes:
    """Return JPEG bytes of the straightened panel with deposits highlighted.

    `panel_bgr` must be the straightened panel (preprocessing.warp_panel output,
    converted to BGR) so the overlay aligns with the region the model analyzes.
    Pure image processing (see module docstring): a large Gaussian blur estimates
    the uneven background illumination; a saturating subtract keeps only pixels
    brighter than that background (deposits show as light specks on the darker
    panel); a threshold isolates them; a direction-aware morphological filter
    removes the straight bus bars and cell edges (see _remove_bus_bars); and the
    result is blended as a semi-transparent highlight over the original panel image.
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

    # Line filter (before the overlay, and before any metric derived from the
    # mask) so the panel's light bus bars are not highlighted/counted as dirt.
    mask = _remove_bus_bars(mask)

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
