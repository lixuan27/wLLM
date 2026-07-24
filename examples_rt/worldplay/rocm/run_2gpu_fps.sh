#!/usr/bin/env bash
# WorldPlay: highest-frame-rate 2-GPU backend (DiT || VAE pipelined) (2 GPUs, CUDA devices 0..1).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.worldplay.backend.launch --variant pipe_dit1_vae1 "$@"
