#!/usr/bin/env bash
# LongLive: 8-GPU backend: DiT SP=4 || 4-GPU tiled VAE, pipelined (8 GPUs, CUDA devices 0..7).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.longlive.backend.launch --variant pipeline_dit4_vae4 "$@"
