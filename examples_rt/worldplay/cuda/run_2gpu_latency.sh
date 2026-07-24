#!/usr/bin/env bash
# WorldPlay: lowest-latency 2-GPU backend (SP=2 DiT + VAE streaming) (2 GPUs, CUDA devices 0..1).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.worldplay.backend.launch --variant sp2_stream "$@"
