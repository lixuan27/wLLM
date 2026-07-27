#!/bin/bash
#SBATCH --job-name=wllm_composite_rollout
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --exclude=server13,server59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Composite graph runtime, measured against real numerics:
#   K concurrent rollout sessions on ONE GPU, expressed as a
#   ComponentGraph + named Walks (Seq / Loop) driven by the executor.
#   1) session isolation  concurrent vs alone must be BIT-IDENTICAL
#   2) step batching      sequential vs fused throughput + parity class
# Both verdicts land in the trace store; a rejected batching row is a
# legitimate outcome and is recorded with its reason.

set -uo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"
cd /public/home/lixuan/lixuan/wllm-infra
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# 64 concurrent walks allocate and free in a ragged order; expandable
# segments keep that from fragmenting the fused-batch arm out of memory.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=/public/home/lixuan/lixuan/wllm-infra/.venv/bin:$PATH
TODAY=$(date +%F)
FAIL=0
step() { echo; echo "== $1"; }

step "composite 0: runtime unit tests (graph / walks / gate / lowering)"
.venv/bin/python -m pytest -q \
  tests/test_composite.py tests/test_composite_walks.py \
  tests/test_composite_rendezvous.py tests/test_composite_lowering.py \
  || FAIL=1

step "composite 1: K concurrent rollout sessions through the runtime"
.venv/bin/python benchmarks/composite_rollout_vjepa2.py \
  | tee /tmp/composite_rollout_$SLURM_JOB_ID.txt
grep -q "COMPOSITE_ROLLOUT_OK" /tmp/composite_rollout_$SLURM_JOB_ID.txt \
  || { echo "composite: benchmark did not report OK"; FAIL=1; }

if [ "$FAIL" -eq 0 ]; then
step "composite 2: record both verdicts in the trace store"
WLLM_CR_TODAY="$TODAY" WLLM_CR_JOB="$SLURM_JOB_ID" \
  .venv/bin/python - << 'PYEOF' || FAIL=1
import glob, json, os, sys
sys.path.insert(0, ".")
from wllm.control.tracestore import Trace, TraceStore

files = sorted(glob.glob("benchmarks/results/composite_rollout_*.json"),
               key=os.path.getmtime)
s = json.load(open(files[-1]))
cfg, arms = s["config"], s["arms"]
iso, bat = s["isolation"], s["batching"]
store = TraceStore(".wllm/traces/beta.jsonl")
workload = (f"{cfg['sessions']} concurrent rollout sessions x "
            f"{cfg['steps']} steps, branch {cfg['branch']}, "
            f"{cfg['frames']}f@{cfg['size']}")
job, today = os.environ["WLLM_CR_JOB"], os.environ["WLLM_CR_TODAY"]
evidence = f"job {job} ({files[-1]})"

isolated = iso["verdict"] == "isolated"
tid_iso = store.append(Trace(
    model="facebook/vjepa2-vitl-fpc64-256", hardware="1xH200",
    runtime="wllm-composite", workload=workload,
    candidate={"pass": "composite_session_isolation",
               "sessions": cfg["sessions"], "branch": cfg["branch"],
               "gpus": 1, "executors": "one shared vs one per session"},
    status="accepted" if isolated else "rejected",
    reason=("every session's rollout output is bit-identical whether it "
            "ran alone or concurrently with the others through one "
            "executor; per-session state does not cross"
            if isolated else
            f"session outputs changed when run concurrently "
            f"(max |diff| {iso['max_abs_over_sessions']:.3e} on "
            f"{iso['worst_session']}); runtime state crosses sessions"),
    metrics={"max_abs_over_sessions": iso["max_abs_over_sessions"],
             "sessions": iso["sessions"],
             "concurrent_steps_per_s": arms["concurrent"]["steps_per_s"],
             "peak_vram_gb": s["peak_vram_gb"]},
    evidence=evidence, recorded=today))

tid_bat = store.append(Trace(
    model="facebook/vjepa2-vitl-fpc64-256", hardware="1xH200",
    runtime="wllm-composite", workload=workload,
    candidate={"pass": "composite_step_batching",
               "sessions": cfg["sessions"], "branch": cfg["branch"],
               "fused_batch": cfg["fused_batch"], "gpus": 1},
    status=bat["status"], reason=bat["reason"],
    metrics={"speedup_vs_sequential": bat["speedup_vs_sequential"],
             "sequential_steps_per_s": bat["sequential_steps_per_s"],
             "batched_steps_per_s": bat["batched_steps_per_s"],
             "parity_class": bat["parity_class"],
             "max_abs": bat["max_abs_over_sessions"],
             "max_rel_l2": bat["max_rel_l2"],
             "budget_rel_l2": bat["budget"]["rel_l2"],
             "peak_vram_gb": s["peak_vram_gb"]},
    evidence=evidence, recorded=today))

print(f"composite traces: isolation={tid_iso} "
      f"({'accepted' if isolated else 'rejected'}) "
      f"batching={tid_bat} ({bat['status']}: {bat['parity_class']}, "
      f"{bat['speedup_vs_sequential']}x)")
print(f"peak VRAM {s['peak_vram_gb']} GB "
      f"(floor {s['vram_floor_gb']} GB, "
      f"{'OK' if s['vram_floor_ok'] else 'LOW'})")
PYEOF
fi

echo; echo "End: $(date)"
if [ "$FAIL" -ne 0 ]; then echo "COMPOSITE_ROLLOUT: BLOCKED"; exit 1; fi
echo "COMPOSITE_ROLLOUT: PASS"
