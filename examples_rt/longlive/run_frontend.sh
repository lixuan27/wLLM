#!/usr/bin/env bash
# LongLive LiveKit frontend. Start a backend first (any script in this
# directory) and wait for the "LongLive backend READY" log line (rank=0),
# then run this. It joins the LiveKit room (default: longlive_room) and
# prints a meet.livekit.io URL to open in your browser; allow the mic and
# narrate -- the generated video updates as you speak new prompts.
#
# One-time setup: copy .env.example to .env at the repo root and fill in your
# LiveKit credentials (see docs/frontends.md).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python -m wllm.apps.longlive.frontend.livekit_frontend "$@"
