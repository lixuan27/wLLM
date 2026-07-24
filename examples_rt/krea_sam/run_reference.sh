#!/usr/bin/env bash
# Krea-Realtime + SAM3: sequential reference backend (SAM then Krea on one GPU) (1 GPU, CUDA device 0).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python -m wllm.apps.krea_sam.backend.launch --variant reference "$@"
