#!/usr/bin/env bash
# Krea-Realtime + SAM3: 4-GPU backend: DiT SP=3 + compiled SAM + frame streaming (4 GPUs, CUDA devices 0..3).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.krea_sam.backend.launch --variant combined_sp3_compile_stream "$@"
