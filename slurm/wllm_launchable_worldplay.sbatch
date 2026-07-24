#!/bin/bash
#SBATCH --job-name=wllm_wp_launchable
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=00:55:00
#SBATCH --exclude=server13,server59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"
cd /public/home/lixuan/lixuan/wllm-infra
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export PATH=/public/home/lixuan/lixuan/wllm-infra/.venv/bin:$PATH
export TRITON_CACHE_DIR=/tmp/wllm_triton_$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/wllm_inductor_$SLURM_JOB_ID
.venv/bin/python tests/launchable_worldplay.py
echo "End: $(date)"
