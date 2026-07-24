#!/usr/bin/env bash
# WorldPlay: lowest-latency 1-GPU backend (per-latent VAE streaming) (1 GPU, CUDA device 0).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.worldplay.backend.launch --variant stream_frames "$@"
