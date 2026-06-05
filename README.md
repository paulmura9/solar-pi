# Solar Pi Gateway

Python edge gateway that runs on a Raspberry Pi 3B and bridges the local device
network and the cloud backend. It is the Raspberry Pi component of LightTrack, a
bachelor's thesis project for monitoring a solar panel. The gateway relays
telemetry and commands between an ESP32 on the local network and an Express
backend in the cloud, and runs the on-device camera pipeline for dirt detection.

## Architecture

The Pi hosts a local Mosquitto MQTT broker (username/password authenticated)
that the ESP32 connects to; this broker is never exposed publicly. This gateway
process connects to the broker as a client and to the Express backend over a
persistent, API-key-authenticated WebSocket (the `/ws/device` endpoint). Sensor
telemetry and ESP32 acknowledgements/events received over MQTT are forwarded up
to Express over the WebSocket; commands coming back from Express over the same
WebSocket are relayed down to the ESP32 over MQTT. The gateway also owns the
camera and runs the dirt-detection pipeline locally on the Pi, uploading images
to Supabase Storage over HTTPS. The Pi has Storage access only and never talks to
the database directly.

A SQLite offline buffer holds messages while the WebSocket is down, the client
reconnects with exponential backoff, and a heartbeat is sent every 30 seconds.

## Camera pipeline

Frames are captured with picamera2 through a single `CameraManager` that returns
one full-resolution BGR frame per capture.

Before any model runs, each frame passes a quality gate (`check_frame_quality`
in `vision.py`). The gate samples a fixed panel region and rejects frames that
are too dark, too bright, not the panel (skin or another object covering the
dark-blue panel), or too low in detail. Rejected frames are reported without a
model prediction.

Dirt detection uses a TFLite model loaded once at startup (via `ai-edge-litert`,
the maintained successor to `tflite-runtime`). Preprocessing applies a fixed
perspective warp that straightens the side-on view of the panel
(`preprocessing.py`), then resizes to 224x224 and normalizes to `[0, 1]`. This
transform must stay identical to the one used to train the model. The model
outputs a softmax over three classes — `clean`, `slightly_dirty`, `dirty` — which
are collapsed into a 0..100 dirt level and a cleanliness percentage.

`surface_analysis.py` produces an auxiliary OpenCV overlay that highlights likely
surface-deposit zones on the straightened panel image. It is a classical
image-processing estimate for the operator and does not feed the model's
decision; the predicted class and percentages come solely from the model.

Captured frames and the overlay are uploaded to Supabase Storage, and the result
metadata is sent to the backend over the WebSocket.

## Tech stack

- Python
- paho-mqtt (`paho-mqtt`) with a local Mosquitto broker
- picamera2
- OpenCV (system package `python3-opencv`)
- TFLite runtime via `ai-edge-litert`
- Supabase Python client (Storage only)
- websockets (persistent client to the Express backend)
- systemd service (`solar-gateway.service`) in production

## Data flow

ESP32 -> MQTT (local) -> this gateway -> WebSocket -> backend

## Running

picamera2 and OpenCV come from the system, so create the virtual environment with
access to system site packages, then install the remaining dependencies:

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

Configuration is centralized in `pi_gateway/config.py`, which reads its values
from the environment (a `.env` file is loaded at startup). The required variables
are `EXPRESS_WS_URL`, `DEVICE_API_KEY`, `SUPABASE_URL`, `SUPABASE_STORAGE_KEY`
and `SUPABASE_STORAGE_BUCKET`; missing required values fail fast at startup. MQTT
settings, the camera/vision tunables and the buffer location have defaults that
can be overridden through the same environment.

Run the gateway with:

```bash
python gateway.py
```

In production the gateway runs as a systemd unit (`solar-gateway.service`), which
starts it on boot and restarts it automatically on failure.

## Constraint: single camera instance

The Pi camera is exclusive — only one process may hold it at a time. Within the
gateway all captures are serialized through one `CameraManager` lock, so the
periodic dirt-detection loop and manual captures never use the camera
concurrently. To collect a dataset with a separate script while the gateway runs,
start the gateway with `VISION_ENABLED=false` so it never opens the camera; every
other path (WebSocket, MQTT, command forwarding, telemetry, heartbeat) keeps
working.
