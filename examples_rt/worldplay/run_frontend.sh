#!/usr/bin/env bash
# WorldPlay web frontend (WebRTC). Start a backend first (any script in this
# directory) and wait for the "WorldPlay backend READY" log line, then run
# this and open http://localhost:8080 in a browser.
#
# WLLM_HOST / WLLM_PORT override the bind address. (HOST/PORT are not
# used because conda activation sets HOST to a compiler triplet, which
# aiohttp would try to resolve as a hostname.)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python -m wllm.apps.worldplay.frontend.server \
    --host "${WLLM_HOST:-0.0.0.0}" \
    --port "${WLLM_PORT:-8080}" \
    "$@"
