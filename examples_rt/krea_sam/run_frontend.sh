#!/usr/bin/env bash
# Krea-Realtime + SAM3 LiveKit frontend. Start a backend first (any script in
# this directory) and wait for the "KreaSAM backend READY" log line, then run
# this. It joins the LiveKit room (default: krea_sam_room) and prints a
# meet.livekit.io URL to open in your browser; allow the webcam there and the
# stylized stream comes back on the frontend's track.
#
# One-time setup: copy .env.example to .env at the repo root and fill in your
# LiveKit credentials (see docs/frontends.md).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python -m wllm.apps.krea_sam.frontend.livekit_frontend "$@"
