"""Temporary diagnostic: measure bus-bar thickness in the surface-overlay mask.

NOT part of the gateway. Its only job is to calibrate the bus-bar removal in
pi_gateway/surface_analysis.py. The current shape filter (a per-contour
bounding-box aspect-ratio test) keeps the panel's bus bars whenever they
interconnect into a single bright lattice, because that lattice is one contour
whose bounding box spans the whole panel (aspect ~1.5, well under the cutoff).
The fix is a morphological opening that erases any structure thinner than its
kernel - but that kernel must be wider than the bus bars and narrower than the
smallest real deposit. This probe measures exactly that, on the SAME straightened
panel and the SAME threshold mask the overlay uses.

It reuses pi_gateway.preprocessing.warp_panel (imports only cv2/numpy - no config
or required-env import) and duplicates the two SURFACE_* mask constants locally,
the same pattern diagnostics/quality_probe.py uses. Keep them in sync with
pi_gateway/config.py while calibrating.

Usage:
    python3 diagnostics/busbar_probe.py                 # capture one frame off the Pi camera
    python3 diagnostics/busbar_probe.py path/to/raw.jpg # or measure a saved full-frame capture

Outputs printed stats AND writes the warped panel, the raw mask, and the mask
after opening at several kernel sizes into diagnostics/busbar_probe_out/ so you
can eyeball which kernel kills the bus bars while keeping the deposits.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pi_gateway.preprocessing import warp_panel

SURFACE_BLUR_KERNEL = 51
SURFACE_DIFF_THRESHOLD = 25
CAPTURE_SIZE = (2304, 1296)
CAMERA_PIXEL_FORMAT = "BGR888"

OPEN_KERNELS = (3, 5, 7, 9, 11, 13)

OUT_DIR = Path(__file__).resolve().parent / "busbar_probe_out"


def _build_mask(panel_bgr: np.ndarray) -> np.ndarray:
    """Exact mask from surface_analysis.build_surface_overlay (before any filter)."""
    gray = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (SURFACE_BLUR_KERNEL, SURFACE_BLUR_KERNEL), 0)
    diff = cv2.subtract(gray, background)
    _, mask = cv2.threshold(diff, SURFACE_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    return mask


def _report(mask: np.ndarray) -> None:
    """Print thickness distribution and an opening-kernel sweep for the mask."""
    total = mask.size
    on = int((mask > 0).sum())
    print(f"mask coverage: {on}/{total} px ({100.0 * on / total:.2f}%)")
    if on == 0:
        print("  empty mask - nothing to measure (check threshold / framing).")
        return

    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    thickness = 2.0 * dt[mask > 0]
    pcts = [50, 75, 90, 95, 99]
    vals = np.percentile(thickness, pcts)
    print("local thickness (px) percentiles over mask pixels:")
    for p, v in zip(pcts, vals):
        print(f"  p{p:<2} = {v:5.1f}")
    print(f"  max  = {thickness.max():5.1f}")

    print("opening-kernel sweep (mask area retained after MORPH_OPEN):")
    for k in OPEN_KERNELS:
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)
        kept = int((opened > 0).sum())
        retained = 100.0 * kept / on if on else 0.0
        print(f"  k={k:<2} -> {kept:7d} px ({retained:5.1f}% kept)")

    print(
        "\nPick the smallest kernel where the bus bars disappear in the saved PNGs\n"
        "but the deposits remain. Bus bars vanish once k exceeds their thickness\n"
        "(see the percentiles); size SURFACE_HLINE_LENGTH / SURFACE_VLINE_LENGTH\n"
        "in pi_gateway/config.py accordingly."
    )


def _save_debug(panel_bgr: np.ndarray, mask: np.ndarray) -> None:
    """Write the warped panel, raw mask, and per-kernel opened masks for review."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_DIR / "panel.png"), panel_bgr)
    cv2.imwrite(str(OUT_DIR / "mask_raw.png"), mask)
    for k in OPEN_KERNELS:
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)
        cv2.imwrite(str(OUT_DIR / f"mask_open_{k}.png"), opened)
    print(f"\nwrote panel + masks to {OUT_DIR}")


def _load_frame_bgr(image_path: Optional[str]) -> np.ndarray:
    """Get one full-frame BGR image: from disk if a path is given, else the camera."""
    if image_path is not None:
        path = Path(image_path)
        if not path.is_file():
            raise SystemExit(f"image not found: {path}")
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"could not decode image (missing or corrupt): {path}")
        return frame

    try:
        from picamera2 import Picamera2
    except Exception as exc:
        raise SystemExit(
            f"no image path given and the camera is unavailable: {exc}\n"
            "pass a saved full-frame capture: python3 diagnostics/busbar_probe.py raw.jpg"
        ) from exc

    try:
        camera = Picamera2()
        camera.configure(
            camera.create_still_configuration(
                main={"size": CAPTURE_SIZE, "format": CAMERA_PIXEL_FORMAT}
            )
        )
        camera.start()
    except Exception as exc:
        raise SystemExit(f"camera start failed: {exc}") from exc
    try:
        frame = camera.capture_array()
    finally:
        camera.stop()
        camera.close()
    if frame is None or frame.size == 0:
        raise SystemExit("camera returned an empty frame.")
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def main() -> None:
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    frame_bgr = _load_frame_bgr(image_path)
    panel_bgr = cv2.cvtColor(warp_panel(frame_bgr), cv2.COLOR_RGB2BGR)
    mask = _build_mask(panel_bgr)
    _report(mask)
    _save_debug(panel_bgr, mask)


if __name__ == "__main__":
    sys.exit(main())
