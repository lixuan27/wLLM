#!/usr/bin/env bash
# LongLive: 4-GPU backend: DiT sequence-parallel and VAE tiled on one group (4 GPUs, CUDA devices 0..3).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.longlive.backend.launch --variant unified_sp4 "$@"
