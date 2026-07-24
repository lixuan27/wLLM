#!/bin/bash
#SBATCH --job-name=wllm_build_sm90_retry1
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=server13,server59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"
ROOT=/public/home/lixuan/lixuan/wllm-infra
cd $ROOT
export PATH=/public/home/lixuan/miniconda3/envs/actiontok/bin:$ROOT/.venv/bin:/usr/local/cuda-12.8/bin:$PATH
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
BUILD=/tmp/wllm_build_sm90_$SLURM_JOB_ID
cmake -S native -B $BUILD -G Ninja \
  -DGPU_ARCH=90 \
  -DFA2_ARCH_NATIVE_ONLY=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE=$ROOT/.venv/bin/python 2>&1 | tail -25
ninja -C $BUILD -j 32 2>&1 | tail -30
echo "== artifacts =="
find $BUILD -name '*.so' -newer $BUILD/CMakeCache.txt -exec ls -la {} \;
mkdir -p native/build_sm90
find $BUILD -name '*.so' -exec cp {} native/build_sm90/ \;
ls -la native/build_sm90/
echo "BUILD_SM90_OK"
echo "End: $(date)"
