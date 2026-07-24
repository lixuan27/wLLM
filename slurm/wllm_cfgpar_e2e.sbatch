#!/bin/bash
#SBATCH --job-name=wllm_wan22_cfgpar_e2e_r3
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=00:55:00
#SBATCH --exclude=server13,server59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"

cd /public/home/lixuan/lixuan/wllm-infra
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH=/public/home/lixuan/lixuan/wllm-infra/.venv/bin:$PATH
export TRITON_CACHE_DIR=/tmp/wllm_triton_$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/wllm_inductor_$SLURM_JOB_ID
export NCCL_DEBUG=WARN TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
export WLLM_CFGPAR_E2E=1

echo "== phase 1: E2E single-GPU reference (sequential branches + decode)"
WLLM_CFGPAR_MODE=ref1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/cfgpar_wan22.py

echo "== phase 2: E2E single-GPU batched CFG + decode"
WLLM_CFGPAR_MODE=batched CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/cfgpar_wan22.py

echo "== phase 3: E2E 2-GPU CFG branch parallel + rank0 decode"
WLLM_CFGPAR_MODE=par2 .venv/bin/python -m torch.distributed.run --standalone \
  --nproc_per_node=2 benchmarks/cfgpar_wan22.py

echo "== phase 4: frame-level exact gate"
.venv/bin/python - << 'PYEOF'
import torch, json
base = "benchmarks/results/wan22_cfgpar_frames_"
ref = torch.load(base + "ref1.pt"); par = torch.load(base + "par2.pt")
bat = torch.load(base + "batched.pt")
def cmp(a, b, name):
    d = (a.to(torch.int16) - b.to(torch.int16)).abs()
    r = {"pair": name, "max_abs": int(d.max()), "mean_abs": float(d.float().mean()),
         "bitexact": bool(torch.equal(a, b))}
    print(json.dumps(r)); return r
r1 = cmp(ref, par, "frames_ref1_vs_par2")
cmp(ref, bat, "frames_ref1_vs_batched")
print("E2E_EXACT_GATE_PASS" if r1["bitexact"] else
      ("E2E_EXACT_GATE_TOL" if r1["max_abs"] <= 1 else "E2E_EXACT_GATE_FAIL"))
PYEOF

echo "End: $(date)"
