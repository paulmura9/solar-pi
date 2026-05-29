#!/usr/bin/env python3
"""LightTrack Pi gateway entrypoint.

Thin launcher kept at the repo root so `python gateway.py` (see run.sh / the
systemd unit) keeps working. All logic lives in the `pi_gateway` package,
organized by responsibility.
"""
import asyncio

from pi_gateway.app import main

if __name__ == "__main__":
    asyncio.run(main())
