#!/usr/bin/env bash
# LongLive: 7-GPU backend (SP=4 DiT + 2-GPU tile VAE + async ASR; recommended) (7 GPUs, CUDA devices 0..6).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.longlive.backend.launch --variant combined_sp4_vae2 "$@"
