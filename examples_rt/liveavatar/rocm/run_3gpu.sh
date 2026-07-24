#!/usr/bin/env bash
# LiveAvatar: 3-GPU backend: chunk streaming plus TTS overlap, lowest latency (3 GPUs, CUDA devices 0..2).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
# Route this app's vLLM engines through AMD's AITER kernels, which is how the
# variant was measured.
export VLLM_ROCM_USE_AITER=1
python -m wllm.apps.liveavatar.backend.launch --variant stream_full "$@"
