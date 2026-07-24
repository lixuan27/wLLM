#!/usr/bin/env bash
# Krea+SAM: 4-GPU backend: sequence-parallel DiT, per-frame emit, compiled SAM (4 GPUs, CUDA devices 0..3).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.krea_sam.backend.launch --variant stream_sp3_compiled "$@"
