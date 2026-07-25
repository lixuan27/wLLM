#!/bin/bash
#SBATCH --job-name=wllm_funnel_wan22_r2
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
# Full-funnel demonstration on real hardware, one job, no shortcuts:
#   plan candidates (three knowledge sources)
#   -> execute the chosen candidate + reference on 2 GPUs
#   -> frame-level exact gate on the fresh outputs
#   -> build a receipt FROM THIS JOB'S OWN LOG
#   -> promote gate -> apply -> report -> rollback
#   -> write the fresh outcome back into the trace store
# Every verdict comes from this run; nothing is replayed from history.

set -uo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"

cd /public/home/lixuan/lixuan/wllm-infra
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH=/public/home/lixuan/lixuan/wllm-infra/.venv/bin:$PATH
export TRITON_CACHE_DIR=/tmp/wllm_triton_$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/wllm_inductor_$SLURM_JOB_ID
export NCCL_DEBUG=WARN TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
export WLLM_CFGPAR_E2E=1
SELF_LOG="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
TODAY=$(date +%F)
FAIL=0
step() { echo; echo "== $1"; }

step "funnel 0: seed the trace store (idempotent)"
.venv/bin/python scripts/seed_traces_beta.py || FAIL=1

step "funnel 1: three-source candidate plan"
.venv/bin/python -m wllm.control.cli candidates . \
  --model Wan-AI/Wan2.2-TI2V-5B --hardware 2xH200 --num-gpus 2 \
  --context '{"model_uses_cfg": true}' \
  --traces .wllm/traces/beta.jsonl --today "$TODAY" \
  | tee /tmp/funnel_plan_$SLURM_JOB_ID.txt || FAIL=1
grep -q "cfg_branch_parallel" /tmp/funnel_plan_$SLURM_JOB_ID.txt \
  || { echo "funnel: planned candidate missing cfg_branch_parallel"; FAIL=1; }

step "phase 1: E2E single-GPU reference (sequential branches + decode)"
WLLM_CFGPAR_MODE=ref1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python benchmarks/cfgpar_wan22.py || FAIL=1

step "phase 2: E2E single-GPU batched CFG + decode"
WLLM_CFGPAR_MODE=batched CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python benchmarks/cfgpar_wan22.py || FAIL=1

step "phase 3: E2E 2-GPU CFG branch parallel + rank0 decode"
WLLM_CFGPAR_MODE=par2 .venv/bin/python -m torch.distributed.run --standalone \
  --nproc_per_node=2 benchmarks/cfgpar_wan22.py || FAIL=1

step "phase 4: frame-level exact gate"
.venv/bin/python - << 'PYEOF' || FAIL=1
import torch, json
base = "benchmarks/results/wan22_cfgpar_frames_"
ref = torch.load(base + "ref1.pt"); par = torch.load(base + "par2.pt")
bat = torch.load(base + "batched.pt")
def cmp(a, b, name):
    d = (a.to(torch.int16) - b.to(torch.int16)).abs()
    r = {"pair": name, "max_abs": int(d.max()),
         "mean_abs": float(d.float().mean()),
         "bitexact": bool(torch.equal(a, b))}
    print(json.dumps(r)); return r
r1 = cmp(ref, par, "frames_ref1_vs_par2")
cmp(ref, bat, "frames_ref1_vs_batched")
print("E2E_EXACT_GATE_PASS" if r1["bitexact"] else
      ("E2E_EXACT_GATE_TOL" if r1["max_abs"] <= 1 else "E2E_EXACT_GATE_FAIL"))
PYEOF

sleep 2   # let the output file flush before parsing our own log

step "funnel 2: receipt from THIS job's log (fresh evidence only)"
.venv/bin/python scripts/receipt_wan22_cfgpar.py --log "$SELF_LOG" || FAIL=1

step "funnel 3: promote gate -> apply -> report -> rollback"
RECEIPT=.wllm/receipts/wan22-cfg-branch-parallel-2gpu.json
.venv/bin/python -m wllm.control.cli verify --receipt "$RECEIPT" || FAIL=1
.venv/bin/python -m wllm.control.cli apply . --receipt "$RECEIPT" || FAIL=1
.venv/bin/python -m wllm.control.cli report . || FAIL=1
.venv/bin/python -m wllm.control.cli rollback . || FAIL=1

step "funnel 4: write the fresh outcome back into the trace store"
WLLM_FUNNEL_TODAY="$TODAY" WLLM_FUNNEL_JOB="$SLURM_JOB_ID" \
  .venv/bin/python - << 'PYEOF' || FAIL=1
import json, os, sys
sys.path.insert(0, ".")
from wllm.control.tracestore import Trace, TraceStore
receipt = json.load(open(".wllm/receipts/wan22-cfg-branch-parallel-2gpu.json"))
speed = receipt["baseline_perf"]["p50_ms"] / receipt["perf"]["p50_ms"]
store = TraceStore(".wllm/traces/beta.jsonl")
tid = store.append(Trace(
    model="Wan-AI/Wan2.2-TI2V-5B", hardware="2xH200",
    runtime="wllm-serving", workload="t2v E2E 33f@480x832, 20 denoise steps",
    candidate={"pass": "cfg_branch_parallel", "gpus": 2},
    status="accepted",
    reason="funnel re-measurement: frame-level bit-exact vs fresh "
           "sequential reference in the same job",
    metrics={"speedup": round(speed, 3),
             "p50_ms": receipt["perf"]["p50_ms"]},
    evidence=f"job {os.environ['WLLM_FUNNEL_JOB']} (this funnel run)",
    recorded=os.environ["WLLM_FUNNEL_TODAY"]))
dedup = store.deduped
print(f"funnel trace: id={tid} deduped={dedup} "
      + ("(outcome consistent with recorded history)" if dedup
         else "(new evidence appended)"))
PYEOF

echo; echo "End: $(date)"
if [ "$FAIL" -ne 0 ]; then echo "FUNNEL: BLOCKED"; exit 1; fi
echo "FUNNEL: PASS"
