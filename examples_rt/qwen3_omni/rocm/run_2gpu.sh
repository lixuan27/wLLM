#!/usr/bin/env bash
# Qwen3-Omni: 2-GPU backend: all three stages streamed (2 GPUs, CUDA devices 0..1).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
# Route this app's vLLM engines through AMD's AITER kernels, which is how the
# variant was measured.
export VLLM_ROCM_USE_AITER=1
python -m wllm.apps.qwen3_omni.backend.launch --variant full_stream "$@"
