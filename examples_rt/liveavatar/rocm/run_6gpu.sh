#!/usr/bin/env bash
# LiveAvatar: 6-GPU backend: the 4 denoise steps pipelined across GPUs (6 GPUs, CUDA devices 0..5).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
# Route this app's vLLM engines through AMD's AITER kernels, which is how the
# variant was measured.
export VLLM_ROCM_USE_AITER=1
python -m wllm.apps.liveavatar.backend.launch --variant denoise_pp4 "$@"
