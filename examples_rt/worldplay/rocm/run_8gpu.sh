#!/usr/bin/env bash
# WorldPlay: 8-GPU backend (SP=4 DiT || 4-way tiled VAE pipelined) (8 GPUs, CUDA devices 0..7).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.worldplay.backend.launch --variant pipe_dit4_vae4 "$@"
