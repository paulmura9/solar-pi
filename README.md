# Solar Pi Gateway

Raspberry Pi gateway that bridges local MQTT (ESP32 communication) with the cloud Express backend via authenticated WebSocket.

## Architecture
ESP32 ──MQTT──► Pi (this) ──WSS──► Express ──HTTPS──► Supabase

## Setup

1. Clone this repo to Pi
2. Create virtual environment:
```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
```
3. Copy `.env.example` to `.env` and fill in real values
4. Create buffer directory:
```bash
   sudo mkdir -p /var/lib/solar-tracker
   sudo chown $USER:$USER /var/lib/solar-tracker
```
5. Run:
```bash
   python gateway.py
```

## Features

- Persistent WebSocket connection with exponential backoff reconnect
- SQLite local buffer for offline resilience
- MQTT bridge for ESP32 communication
- Structured JSON logging
- Heartbeat every 30 seconds
- Computer vision pipeline for dirt detection (planned)

## Dataset collection

Interactive tool to capture labelled dirt-detection images on the Pi. Run it on the device and follow the prompts (session id + class, then ENTER to capture / `q` to quit). Images are saved under `dataset/raw/{session_id}/{class}/` and logged to `dataset/manifest.csv`:
```bash
   python dataset/capture_dataset.py
```

The Pi camera is exclusive (one process at a time). To collect a dataset while the gateway runs, start the gateway with `VISION_ENABLED=false` so it never opens the camera (everything else — WS, MQTT, command forwarding, telemetry, heartbeat — keeps working): `VISION_ENABLED=false python gateway.py`.
