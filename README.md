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
