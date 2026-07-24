#!/bin/bash
#SBATCH --job-name=wllm_wan22_cfgpar_r2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --time=00:50:00
#SBATCH --exclude=server13,server59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"

cd /public/home/lixuan/lixuan/wllm-infra
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH=/public/home/lixuan/lixuan/wllm-infra/.venv/bin:$PATH
export NCCL_DEBUG=WARN TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

echo "== phase 1: single-GPU like-for-like reference (sequential branches)"
WLLM_CFGPAR_MODE=ref1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/cfgpar_wan22.py

echo "== phase 2: single-GPU batched CFG (pipeline-native shape)"
WLLM_CFGPAR_MODE=batched CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/cfgpar_wan22.py

echo "== phase 3: 2-GPU CFG branch parallel"
WLLM_CFGPAR_MODE=par2 .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 \
  benchmarks/cfgpar_wan22.py

echo "== phase 4: exact gate (latent compare)"
.venv/bin/python - << 'PYEOF'
import torch, json
base = "benchmarks/results/wan22_cfgpar_latent_"
ref = torch.load(base + "ref1.pt"); par = torch.load(base + "par2.pt")
bat = torch.load(base + "batched.pt")
def cmp(a, b, name):
    d = (a - b).abs()
    r = {"pair": name, "max_abs": float(d.max()), "mean_abs": float(d.mean()),
         "allclose_1e-5": bool(torch.allclose(a, b, atol=1e-5, rtol=0)),
         "bitexact": bool(torch.equal(a, b))}
    print(json.dumps(r)); return r
r1 = cmp(ref, par, "ref1_vs_par2")
r2 = cmp(ref, bat, "ref1_vs_batched")
print("EXACT_GATE_PASS" if r1["allclose_1e-5"] else "EXACT_GATE_FAIL")
PYEOF

echo "End: $(date)"
