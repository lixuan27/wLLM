#!/usr/bin/env bash
# WorldPlay: highest-frame-rate 4-GPU backend (SP=2 DiT || tiled VAE pipelined) (4 GPUs, CUDA devices 0..3).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.worldplay.backend.launch --variant pipe_dit2_vae2 "$@"
