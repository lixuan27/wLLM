#!/usr/bin/env bash
# Krea+SAM: 5-GPU backend: sequence-parallel DiT with a split VAE stage, smoothest output (5 GPUs, CUDA devices 0..4).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
python -m wllm.apps.krea_sam.backend.launch --variant sp3_vae_split_compiled "$@"
