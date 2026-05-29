"""Camera sanity check for headless Pi.

Captures one frame from the IMX708 sensor, saves it as JPEG,
and prints basic image stats useful for verifying exposure.

Note on color format: picamera2's format strings ("RGB888" / "BGR888")
do not reliably match the channel ordering of the returned numpy array
on little-endian systems like the Raspberry Pi. Empirically, the array
comes out in RGB order, so an explicit RGB -> BGR conversion is required
before passing the frame to cv2.imwrite (which expects BGR).
"""
from pathlib import Path
import sys

import cv2
from picamera2 import Picamera2


OUTPUT_PATH = Path(__file__).parent / "camera_test.jpg"
CAPTURE_RESOLUTION = (2304, 1296)


def main() -> int:
    picam = Picamera2()
    config = picam.create_still_configuration(
        main={"size": CAPTURE_RESOLUTION, "format": "BGR888"}
    )
    picam.configure(config)
    picam.start()
    try:
        frame = picam.capture_array()
    finally:
        picam.stop()
        picam.close()

    # Convert from picamera2's RGB-ordered array to BGR for OpenCV.
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    h, w, _ = frame_bgr.shape
    print(f"Resolution: {w}x{h}")
    print(f"Mean brightness (0-255): {float(frame_bgr.mean()):.1f}")

    cv2.imwrite(str(OUTPUT_PATH), frame_bgr)
    print(f"Saved: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())