#!/usr/bin/env bash
# Krea-Realtime + SAM3: 3-GPU backend: VAE stage and DiT stage pipelined + SAM GPU (3 GPUs, CUDA devices 0..2).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.krea_sam.backend.launch --variant krea_vae_dit_split "$@"
