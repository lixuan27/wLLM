#!/usr/bin/env bash
# LiveAvatar: 8-GPU backend: 2-stage sequence-parallel denoise pipeline driven from the TTS stream (8 GPUs, CUDA devices 0..7).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
# Route this app's vLLM engines through AMD's AITER kernels, which is how the
# variant was measured.
export VLLM_ROCM_USE_AITER=1
python -m wllm.apps.liveavatar.backend.launch --variant stream_pp2_sp3 "$@"
