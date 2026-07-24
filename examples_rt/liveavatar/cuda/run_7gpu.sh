#!/usr/bin/env bash
# LiveAvatar: 7-GPU backend: streaming + DiT+VAE pipeline, LLM+TTS shared; needs GPU P2P (7 GPUs, CUDA devices 0..6).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.liveavatar.backend.launch --variant combined_stream_pp_vae "$@"
