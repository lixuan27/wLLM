#!/usr/bin/env bash
# Krea-Realtime + SAM3: 2-GPU backend: SAM compiled on its own GPU + frame streaming (2 GPUs, CUDA devices 0..1).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.krea_sam.backend.launch --variant combined_compile_stream "$@"
