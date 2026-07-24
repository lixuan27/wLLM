#!/usr/bin/env bash
# LongLive: 4-GPU backend (DiT SP=4, tiled VAE; real-time fps, best latency) (4 GPUs, CUDA devices 0..3).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.longlive.backend.launch --variant dit_sp4 "$@"
