#!/usr/bin/env bash
# LongLive: 2-GPU backend, highest fps + bit-exact (DiT/VAE cross-chunk pipeline) (2 GPUs, CUDA devices 0..1).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.longlive.backend.launch --variant vae_decouple "$@"
