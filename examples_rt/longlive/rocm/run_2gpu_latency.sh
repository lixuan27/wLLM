#!/usr/bin/env bash
# LongLive: lowest-latency 2-GPU backend (sequence-parallel DiT) (2 GPUs, CUDA devices 0..1).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.longlive.backend.launch --variant dit_sp2 "$@"
