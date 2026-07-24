#!/bin/bash
#SBATCH --job-name=wllm_ll_launchable_r12
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=00:40:00
#SBATCH --exclude=server13,server59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"

cd /public/home/lixuan/lixuan/wllm-infra
export TOKENIZERS_PARALLELISM=false
export PATH=/public/home/lixuan/lixuan/wllm-infra/.venv/bin:$PATH
# ASR + Wan components resolve from local disk; keep hub offline
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/wllm_triton_$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/wllm_inductor_$SLURM_JOB_ID

.venv/bin/python tests/launchable_longlive.py

echo "End: $(date)"
