#!/usr/bin/env bash
# WorldPlay: lowest-latency 4-GPU backend (SP=4 DiT + VAE streaming) (4 GPUs, CUDA devices 0..3).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.worldplay.backend.launch --variant sp4_stream "$@"
