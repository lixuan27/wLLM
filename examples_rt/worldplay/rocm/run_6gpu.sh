#!/usr/bin/env bash
# WorldPlay: 6-GPU backend (SP=4 DiT || 2-way tiled VAE pipelined) (6 GPUs, CUDA devices 0..5).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.worldplay.backend.launch --variant pipe_dit4_vae2 "$@"
