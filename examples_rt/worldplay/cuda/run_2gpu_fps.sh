#!/usr/bin/env bash
# WorldPlay: highest-FPS 2-GPU backend (DiT/VAE stage pipeline) (2 GPUs, CUDA devices 0..1).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.worldplay.backend.launch --variant stage_split_2g "$@"
