#!/usr/bin/env bash
# WorldPlay: 6-GPU backend (SP=4 DiT stage + 2-GPU VAE stage) (6 GPUs, CUDA devices 0..5).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.worldplay.backend.launch --variant stage_split_sp_6g "$@"
