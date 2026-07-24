#!/usr/bin/env bash
# WorldPlay: 7-GPU backend (SP=4 DiT stage + 3-GPU VAE stage) (7 GPUs, CUDA devices 0..6).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.worldplay.backend.launch --variant stage_split_sp_7g "$@"
