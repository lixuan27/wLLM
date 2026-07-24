#!/usr/bin/env bash
# LiveAvatar: 2-GPU streaming backend (low latency) (2 GPUs, CUDA devices 0..1).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.liveavatar.backend.launch --variant stream_app "$@"
