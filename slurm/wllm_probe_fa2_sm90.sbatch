#!/bin/bash
#SBATCH --job-name=wllm_probe_fa2_sm90
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --exclude=server13,server59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"
cd /public/home/lixuan/lixuan/wllm-infra
export PATH=/public/home/lixuan/lixuan/wllm-infra/.venv/bin:$PATH
.venv/bin/python tests/probe_fa2_sm90.py
echo "End: $(date)"
