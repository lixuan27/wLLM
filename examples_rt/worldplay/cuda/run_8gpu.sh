#!/usr/bin/env bash
# WorldPlay: 8-GPU backend (SP=4 DiT stage + 4-GPU VAE stage) (8 GPUs, CUDA devices 0..7).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.worldplay.backend.launch --variant stage_split_sp_8g "$@"
