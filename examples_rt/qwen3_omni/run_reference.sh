#!/usr/bin/env bash
# Qwen3-Omni: sequential reference backend (burst output) (2 GPUs, CUDA devices 0..1).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
python -m wllm.apps.qwen3_omni.backend.launch --variant reference "$@"
