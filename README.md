# LightTrack — Raspberry Pi Edge Gateway (`solar-pi`)

## 1. Project

`solar-pi` is the Raspberry Pi edge gateway of **LightTrack**, a dual-axis solar
tracker with camera-based dirt detection. It is written in Python and runs on a
Raspberry Pi 3B.

LightTrack is a four-layer system:

1. an **ESP32** that drives the two servos and reads the sensors (firmware),
2. **this Raspberry Pi gateway** (the edge layer),
3. an **Express + Supabase** cloud backend, and
4. a **Next.js** web dashboard.

The gateway sits in the middle and connects the local hardware to the cloud. On
one side it talks to the ESP32 over a **local MQTT broker** (Mosquitto, running on
the Pi, never exposed to the internet). On the other side it keeps a persistent,
API-key-authenticated **WebSocket** connection to the Express backend. It also
owns the Pi camera and runs the **dirt-detection pipeline locally**, uploading the
resulting images to Supabase Storage over HTTPS.

**Main features**

- **MQTT ↔ WebSocket bridge.** Telemetry, ESP32 acknowledgements and device
  events arrive over MQTT and are forwarded up to the backend over the WebSocket.
  Commands coming down from the backend over the WebSocket are relayed to the
  ESP32 over MQTT.
- **SQLite offline buffer with reconnect and heartbeat.** When the WebSocket is
  down, outgoing messages are stored in a local SQLite buffer instead of being
  lost. The client reconnects automatically with exponential backoff, flushes the
  buffered messages once it is back online, and sends a heartbeat every 30
  seconds so the backend knows the Pi is alive.
- **Camera dirt-detection pipeline**, running entirely on the Pi:
  - a **quality gate** rejects unusable frames before the model runs (too dark,
    too bright, an object covering the panel, or too little detail);
  - a **TFLite model** classifies the panel into three classes —
    `clean`, `slightly_dirty`, `dirty` — which are turned into a dirt level and a
    cleanliness percentage;
  - **perspective-warp preprocessing** straightens the side-on view of the panel
    before inference, using the exact same transform the model was trained with;
  - a **surface overlay** (classical image processing) highlights likely dirt
    zones on the straightened image, as a visual aid for the operator — it does
    not affect the model's decision;
  - the captured frame and the overlay are **uploaded to Supabase Storage**, and
    only the result metadata is sent to the backend over the WebSocket.

The dirt-detection loop and any manual capture share a single camera through one
lock, so the camera is never opened by two code paths at the same time.

## 2. Deliverables / Repository

- **Repository:** `<https://gitlab.upt.ro/...>`

This repository is the **full source code** of the gateway. It contains **no
compiled binaries** and no build artifacts. Python is an interpreted language, so
there is **no compilation step** — the code runs directly.

The following are intentionally **not committed** to the repository (they are
generated or provided on the Pi):

- `venv/` — the Python virtual environment (created during installation),
- `.env` — the local configuration file with the secrets,
- `models/*.tflite` — the trained dirt-detection model.

## 3. Requirements / Dependencies

**Python.** Python 3, using the **system Python 3 that ships with Raspberry Pi
OS**. This matters because two required libraries (see below) are installed as
system packages and are built against that specific interpreter.

**System packages (installed with `apt`).** Two dependencies are **not available
on PyPI** and must come from the Raspberry Pi OS package manager:

- `python3-opencv` — OpenCV for Python,
- `python3-picamera2` — the `picamera2` camera library.

Install them with:

```bash
sudo apt update
sudo apt install python3-opencv python3-picamera2
```

(`numpy` is pulled in together with these system packages.)

**Python packages (installed with `pip`).** From `requirements.txt`:

- `paho-mqtt` — MQTT client for the local broker,
- `supabase` — Supabase client (used only for Storage uploads),
- `python-dotenv` — loads the `.env` configuration file,
- `websockets` — the persistent client to the Express backend,
- `ai-edge-litert` — the TFLite runtime used for dirt-detection inference (the
  maintained successor to `tflite-runtime`).

Test-only tools are kept separately in `requirements-dev.txt` (`pytest`,
`pytest-asyncio`). They are **not** needed to run the gateway and are **not**
installed on the Pi.

## 4. Installation

Run these steps on the Raspberry Pi, after installing the `apt` packages from
Section 3.

Create the virtual environment **with `--system-site-packages`**. This flag is
required so the environment can see the system-installed `picamera2` and OpenCV,
which cannot be installed through `pip`:

```bash
python3 -m venv --system-site-packages venv
```

Activate the environment:

```bash
source venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## 5. Configuration

All configuration is read from environment variables at startup by
`pi_gateway/config.py`, which loads a **`.env` file** placed in the project root.

The **required** variables have no defaults — if any of them is missing, the
gateway stops immediately at startup with a clear error:

```ini
# Required — the gateway will not start without these
EXPRESS_WS_URL=
DEVICE_API_KEY=
SUPABASE_URL=
SUPABASE_STORAGE_KEY=
SUPABASE_STORAGE_BUCKET=
```

The **optional** variables all have sensible defaults and only need to be set to
override them:

```ini
# Optional — shown with their default values
DEVICE_ID=raspberry-pi-001
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883
MQTT_USERNAME=
MQTT_PASSWORD=
BUFFER_DB_PATH=/var/lib/solar-tracker/buffer.db
BUFFER_MAX_AGE_HOURS=24
VISION_ENABLED=true
VISION_CAPTURE_INTERVAL_S=1800
DIRT_MODEL_PATH=models/dirt_detection.tflite
```

> Fill in your own values for the required variables. Never commit the `.env`
> file — it is ignored by Git.

## 6. Run / Launch

With the virtual environment active, start the gateway from the project root:

```bash
python gateway.py
```

A convenience script, `run.sh`, does the same thing using the environment's
Python directly.

**In production**, the gateway runs as a **systemd service** named
`solar-gateway.service`. systemd starts it automatically on boot and restarts it
if it ever exits, so the Pi keeps bridging the ESP32 and the cloud without manual
intervention. Typical operator commands:

```bash
sudo systemctl restart solar-gateway   # restart the service
journalctl -u solar-gateway -f         # follow the live logs
```

**Freeing the camera for dataset collection.** Set `VISION_ENABLED=false` to start
the gateway **without opening the camera**. The dirt-detection loop is then
disabled, so a separate script can use the camera to collect training images,
while everything else — the WebSocket link, MQTT bridging, command forwarding,
telemetry and heartbeat — keeps working normally.
