"""LightTrack Raspberry Pi gateway.

Edge service bridging the local ESP32 (MQTT) and the Express cloud backend
(authenticated WebSocket). Organized by responsibility: config, logging,
protocol, offline buffer, telemetry validation, WebSocket client, MQTT bridge,
Supabase Storage client, camera manager, CAPTURE_IMAGE handler, dispatcher, and
the SolarGateway orchestrator.
"""
