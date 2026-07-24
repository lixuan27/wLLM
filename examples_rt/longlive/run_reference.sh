#!/usr/bin/env bash
# LongLive: sequential reference backend (1 GPU, CUDA device 0).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python -m wllm.apps.longlive.backend.launch --variant reference "$@"
