#!/usr/bin/env bash
# Krea-Realtime + SAM3: 5-GPU backend: dedicated VAE GPU + DiT SP=3 + compiled SAM + streaming (5 GPUs, CUDA devices 0..4).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.krea_sam.backend.launch --variant combined_sp3_vae_split_compile_stream "$@"
