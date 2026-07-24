#!/bin/bash
#SBATCH --job-name=wllm_qwen3vl_opt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:35:00
#SBATCH --exclude=server13,server59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

cd /public/home/lixuan/lixuan/wllm-infra
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR=/tmp/wllm_inductor_$SLURM_JOB_ID

.venv/bin/python benchmarks/baseline_qwen3vl.py

echo "End time: $(date)"
