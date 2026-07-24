#!/bin/bash
#SBATCH --job-name=wllm_libero_bridge
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --exclude=server13,server59,server45
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"

cd /public/home/lixuan/lixuan/wllm-infra
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export WLLM_VLA_SOCK=/tmp/wllm_vla_$SLURM_JOB_ID.sock

SRV_PY=/public/home/lixuan/miniconda3/envs/openvla/bin/python
CLI_PY=/public/home/lixuan/miniconda3/envs/xactorwm/bin/python

run_arm () {
  local dtype=$1
  echo "== arm: $dtype"
  WLLM_VLA_DTYPE=$dtype $SRV_PY benchmarks/vla_bridge/action_server.py \
      > logs/vla_server_${dtype}_$SLURM_JOB_ID.log 2>&1 &
  local srv_pid=$!
  for i in $(seq 1 120); do
    [ -S "$WLLM_VLA_SOCK" ] && break
    if ! kill -0 $srv_pid 2>/dev/null; then
      echo "SERVER_DIED_$dtype"; tail -5 logs/vla_server_${dtype}_$SLURM_JOB_ID.log; return 1
    fi
    sleep 5
  done
  WLLM_VLA_DTYPE=$dtype $CLI_PY benchmarks/vla_bridge/libero_client.py \
    || { echo "CLIENT_FAIL_$dtype"; kill $srv_pid 2>/dev/null; return 1; }
  wait $srv_pid 2>/dev/null || true
  rm -f "$WLLM_VLA_SOCK"
}

run_arm float32
run_arm bfloat16

echo "BRIDGE_BOTH_ARMS_DONE"
echo "End: $(date)"
