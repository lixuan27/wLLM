#!/bin/bash
#SBATCH --job-name=wllm_convert_worldplay
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --exclude=server13,server59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"
cd /public/home/lixuan/lixuan/wllm-infra
export PATH=/public/home/lixuan/lixuan/wllm-infra/.venv/bin:$PATH
.venv/bin/python tests/_convert_worldplay.py
echo "End: $(date)"
