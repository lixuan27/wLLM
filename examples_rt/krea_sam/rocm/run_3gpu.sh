#!/usr/bin/env bash
# Krea+SAM: 3-GPU backend: Krea pipelined across 2 GPUs, smooth continuous output (3 GPUs, CUDA devices 0..2).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.krea_sam.backend.launch --variant krea_pipeline "$@"
