#!/usr/bin/env bash
# WorldPlay: lowest-latency 2-GPU backend (SP=2 DiT + tiled VAE + streaming) (2 GPUs, CUDA devices 0..1).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.worldplay.backend.launch --variant stream_colocated_sp2 "$@"
