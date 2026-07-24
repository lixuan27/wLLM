#!/usr/bin/env bash
# LiveAvatar: 5-GPU backend: sequence-parallel DiT, first variant to sustain real-time (5 GPUs, CUDA devices 0..4).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
# Route this app's vLLM engines through AMD's AITER kernels, which is how the
# variant was measured.
export VLLM_ROCM_USE_AITER=1
python -m wllm.apps.liveavatar.backend.launch --variant dit_sp3 "$@"
