#!/usr/bin/env bash
# LongLive: 6-GPU backend (SP=4 DiT + decoupled VAE + async ASR) (6 GPUs, CUDA devices 0..5).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.longlive.backend.launch --variant combined_sp4_decouple "$@"
