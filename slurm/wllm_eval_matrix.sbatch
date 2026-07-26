#!/bin/bash
#SBATCH --job-name=wllm_eval_matrix_r5
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=00:50:00
#SBATCH --exclude=server13,server59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Evaluation matrix: fresh re-measurements of the two recorded non-video
# instances (VLM 8B, VLA 7B) on one GPU, written back into the trace
# store as dated evidence, then a three-source candidate plan for each.
# Honest by construction: below-target outcomes are recorded as rejected.

set -uo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"
cd /public/home/lixuan/lixuan/wllm-infra
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH=/public/home/lixuan/lixuan/wllm-infra/.venv/bin:$PATH
export TRITON_CACHE_DIR=/tmp/wllm_triton_$SLURM_JOB_ID
export TORCHINDUCTOR_CACHE_DIR=/tmp/wllm_inductor_$SLURM_JOB_ID
TODAY=$(date +%F)
FAIL=0
step() { echo; echo "== $1"; }

step "eval 0: seed the trace store (idempotent)"
.venv/bin/python scripts/seed_traces_beta.py || FAIL=1

step "eval 1: VLM 8B sweep (AR decode, static KV, exact token gate)"
.venv/bin/python benchmarks/baseline_qwen3vl.py | tee /tmp/eval_qwen3vl_$SLURM_JOB_ID.txt || FAIL=1
grep -q "QWEN3VL_OK" /tmp/eval_qwen3vl_$SLURM_JOB_ID.txt \
  || { echo "eval: VLM sweep below target or failed"; FAIL=1; }

step "eval 2: VLA 7B sweep (single-action latency, native precision)"
.venv/bin/python benchmarks/baseline_openvla.py | tee /tmp/eval_openvla_$SLURM_JOB_ID.txt || FAIL=1
grep -q "OPENVLA_OK" /tmp/eval_openvla_$SLURM_JOB_ID.txt \
  || { echo "eval: VLA sweep below target or failed"; FAIL=1; }

step "eval 3: write fresh dated outcomes into the trace store"
WLLM_EVAL_TODAY="$TODAY" WLLM_EVAL_JOB="$SLURM_JOB_ID" \
  .venv/bin/python - << 'PYEOF' || FAIL=1
import glob, json, os, sys
sys.path.insert(0, ".")
from wllm.control.tracestore import Trace, TraceStore

def latest(pattern):
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        raise SystemExit(f"no summary file matches {pattern}")
    return json.load(open(files[-1])), files[-1]

store = TraceStore(".wllm/traces/beta.jsonl")
today = os.environ["WLLM_EVAL_TODAY"]
job = os.environ["WLLM_EVAL_JOB"]
for model, runtime, workload, pattern in [
    ("Qwen/Qwen3-VL-8B-Instruct", "wllm-serving",
     "AR decode, 128 new tokens", "benchmarks/results/qwen3vl_opt_*.json"),
    ("openvla/openvla-7b", "wllm-native",
     "single-action prediction", "benchmarks/results/openvla_opt_*.json"),
]:
    summary, path = latest(pattern)
    speed = float(summary["speedup"])
    met = bool(summary.get("target_met_1p3x"))
    tid = store.append(Trace(
        model=model, hardware="1xH200", runtime=runtime, workload=workload,
        candidate={"pass": str(summary["best_plan"]), "gpus": 1},
        status="accepted" if met else "rejected",
        reason=("fresh re-measurement met the 1.3x target"
                if met else
                f"fresh re-measurement below 1.3x target "
                f"(speedup {speed:.2f}x)"),
        metrics={"speedup": round(speed, 3),
                 "best_ms": summary.get("best_ms")},
        evidence=f"job {job} ({path})", recorded=today))
    print(f"eval trace: {model} -> {summary['best_plan']} "
          f"{speed:.2f}x id={tid}")
print(f"store now holds {len(store.all())} traces "
      f"(deduped this run: {store.deduped})")
PYEOF

step "eval 4: three-source candidate plans for both instances"
.venv/bin/python -m wllm.control.cli candidates . \
  --model Qwen/Qwen3-VL-8B-Instruct --hardware 1xH200 --num-gpus 1 \
  --traces .wllm/traces/beta.jsonl --today "$TODAY" || FAIL=1
.venv/bin/python -m wllm.control.cli candidates . \
  --model openvla/openvla-7b --hardware 1xH200 --num-gpus 1 \
  --traces .wllm/traces/beta.jsonl --today "$TODAY" || FAIL=1

echo; echo "End: $(date)"
if [ "$FAIL" -ne 0 ]; then echo "EVAL_MATRIX: BLOCKED"; exit 1; fi
echo "EVAL_MATRIX: PASS"
