#!/usr/bin/env bash
# LiveAvatar: sequential reference backend (2 GPUs, CUDA devices 0..1).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python -m wllm.apps.liveavatar.backend.launch --variant reference "$@"
