#!/usr/bin/env bash
# Qwen3-Omni: 3-GPU backend: streamed stages on dedicated GPUs with a tuned vocode schedule (3 GPUs, CUDA devices 0..2).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
# Route this app's vLLM engines through AMD's AITER kernels, which is how the
# variant was measured.
export VLLM_ROCM_USE_AITER=1
python -m wllm.apps.qwen3_omni.backend.launch --variant full_stream_tuned_chunks "$@"
