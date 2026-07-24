#!/usr/bin/env bash
# Qwen3-Omni: 2-GPU backend: full streaming + bounded vocoder context + threaded pump (2 GPUs, CUDA devices 0..1).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.qwen3_omni.backend.launch --variant stream_full_windowed_threaded_2gpu "$@"
