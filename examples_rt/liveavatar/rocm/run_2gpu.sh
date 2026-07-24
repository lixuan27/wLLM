#!/usr/bin/env bash
# LiveAvatar: 2-GPU backend: emit each chunk's frames as produced (2 GPUs, CUDA devices 0..1).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
# Route this app's vLLM engines through AMD's AITER kernels, which is how the
# variant was measured.
export VLLM_ROCM_USE_AITER=1
python -m wllm.apps.liveavatar.backend.launch --variant stream_liveavatar "$@"
