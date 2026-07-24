#!/usr/bin/env bash
# LiveAvatar: 8-GPU backend: streaming + DiT+VAE pipeline, LLM and TTS split; needs GPU P2P (8 GPUs, CUDA devices 0..7).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.liveavatar.backend.launch --variant combined_stream_pp_vae_split "$@"
