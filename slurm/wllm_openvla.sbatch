#!/bin/bash
#SBATCH --job-name=wllm_openvla_opt_r3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=00:45:00
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

# own-env worker: this model runs in its dedicated environment
/public/home/lixuan/miniconda3/envs/openvla/bin/python benchmarks/baseline_openvla.py

echo "End time: $(date)"
