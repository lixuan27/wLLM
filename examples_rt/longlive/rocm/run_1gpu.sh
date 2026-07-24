#!/usr/bin/env bash
# LongLive: 1-GPU backend: decode the first frame before the cache-write pass (1 GPU, CUDA device 0).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.longlive.backend.launch --variant decode_first "$@"
