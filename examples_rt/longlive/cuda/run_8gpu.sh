#!/usr/bin/env bash
# LongLive: 8-GPU backend (SP=4 DiT + 3-GPU tile VAE + async ASR; max throughput) (8 GPUs, CUDA devices 0..7).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.longlive.backend.launch --variant combined_sp4_vae3 "$@"
