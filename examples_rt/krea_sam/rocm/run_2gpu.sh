#!/usr/bin/env bash
# Krea+SAM: 2-GPU backend: per-frame emit with SAM decoupled and compiled (2 GPUs, CUDA devices 0..1).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.krea_sam.backend.launch --variant frame_stream_compiled "$@"
