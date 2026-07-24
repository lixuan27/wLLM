#!/usr/bin/env bash
# LiveAvatar: 6-GPU backend: streaming + DiT step pipeline (low latency, real-time fps) (6 GPUs, CUDA devices 0..5).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.liveavatar.backend.launch --variant combined_stream_pp "$@"
