#!/usr/bin/env bash
# LiveAvatar LiveKit frontend. Start a backend first (any script in this
# directory) and wait for the "LiveAvatar backend READY" log line, then run
# this. It joins the LiveKit room (default: liveavatar_room) and prints a
# meet.livekit.io URL to open in your browser.
#
# One-time setup: copy .env.example to .env at the repo root and fill in your
# LiveKit credentials (see docs/frontends.md).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python -m wllm.apps.liveavatar.frontend.livekit_frontend "$@"
