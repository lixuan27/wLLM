#!/usr/bin/env bash
# Qwen3-Omni: 3-GPU backend: dedicated GPU per stage + full streaming + threaded pump (3 GPUs, CUDA devices 0..2).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.qwen3_omni.backend.launch --variant stream_full_windowed_threaded "$@"
