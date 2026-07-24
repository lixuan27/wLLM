#!/usr/bin/env bash
# WorldPlay: highest-FPS 4-GPU backend (SP=2 DiT stage + 2-GPU VAE stage) (4 GPUs, CUDA devices 0..3).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.worldplay.backend.launch --variant stage_split_sp_4g "$@"
