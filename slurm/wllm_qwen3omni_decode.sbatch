#!/bin/bash
#SBATCH --job-name=wllm_qwen3omni_decode
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --exclude=server13,server59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Omni thinker AR-decode optimization sweep: does any exact-quality pass
# actually beat the reference leg on this 30B-A3B MoE thinker? The model
# is loaded ONCE and every leg runs against it, so the ~2 min load is
# paid a single time and all legs share one reference.
#
# A "no exact win" outcome is a PASS: the sweep succeeded, and the
# rejected candidates are written to the trace store WITH reasons so no
# future planner re-explores them blind. Only a broken sweep is BLOCKED.

set -uo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"
cd /public/home/lixuan/lixuan/wllm-infra
export WLLM_Q3O_PATH=/public/home/lixuan/lixuan/pretrained-model/Qwen3-Omni-30B-A3B-Instruct
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH=/public/home/lixuan/lixuan/wllm-infra/.venv/bin:$PATH
# 55 min of sweep leaves >30 min of headroom under --time, so a slow leg
# degrades into an honest "wall budget exceeded" row instead of a TIMEOUT
# that would leave the whole run unreported.
export WLLM_Q3O_BUDGET_S=3300
TODAY=$(date +%F)
FAIL=0
step() { echo; echo "== $1"; }

step "q3o-decode 1: preflight (weights present, no re-download)"
if [ ! -f "$WLLM_Q3O_PATH/model.safetensors.index.json" ]; then
  echo "q3o-decode: checkpoint index missing at $WLLM_Q3O_PATH"; FAIL=1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

step "q3o-decode 2: AR decode sweep (single load, all legs, exact gate)"
.venv/bin/python benchmarks/decode_qwen3omni.py \
  | tee /tmp/q3o_decode_$SLURM_JOB_ID.txt
grep -q "Q3O_DECODE_SWEEP_OK" /tmp/q3o_decode_$SLURM_JOB_ID.txt \
  || { echo "q3o-decode: sweep did not reach its OK marker"; FAIL=1; }

step "q3o-decode 3: record every leg into the trace store"
# Accepted AND rejected AND failed legs all land in the store. A
# rejection without a reason is worthless to a future planner, so the
# trace store's own validation (reason mandatory for rejected/failed) is
# the gate here -- an append that fails closed fails this job.
if [ "$FAIL" -eq 0 ]; then
  WLLM_Q3O_TODAY="$TODAY" WLLM_Q3O_JOB="$SLURM_JOB_ID" \
    .venv/bin/python - << 'PYEOF' || FAIL=1
import glob, json, os, sys
sys.path.insert(0, ".")
from wllm.control.tracestore import Trace, TraceStore

files = sorted(glob.glob("benchmarks/results/qwen3omni_decode_*.json"),
               key=os.path.getmtime)
if not files:
    raise SystemExit("no decode summary written; nothing to record")
summary = json.load(open(files[-1]))
store = TraceStore(".wllm/traces/beta.jsonl")
counts = {"accepted": 0, "rejected": 0, "failed": 0}
for tr in summary["traces"]:
    tid = store.append(Trace(
        model=summary["model"], hardware="1xH200", runtime="torch-local",
        workload=tr["workload"], candidate=tr["candidate"],
        status=tr["status"], reason=tr["reason"], metrics=tr["metrics"],
        evidence=f"job {os.environ['WLLM_Q3O_JOB']} ({files[-1]})",
        recorded=os.environ["WLLM_Q3O_TODAY"]))
    counts[tr["status"]] += 1
    print(f"  {tr['status']:<9} {tr['candidate']['pass']:<28} "
          f"n={tr['candidate']['new_tokens']:<4} id={tid}")
print(f"q3o-decode traces: {counts} (deduped {store.deduped})")
print(f"q3o-decode headline: any_exact_win={summary['any_exact_win']} "
      f"accepted={summary['accepted_optimizations']}")
PYEOF
else
  echo "q3o-decode: skipping trace append because the sweep did not pass"
fi

step "q3o-decode 4: profile pack still validates and is fresh"
.venv/bin/python tests/test_profiles.py | tail -n 3 \
  | tee /tmp/q3o_prof_$SLURM_JOB_ID.txt
grep -q "ALL PASS" /tmp/q3o_prof_$SLURM_JOB_ID.txt \
  || { echo "q3o-decode: profile pack test not green"; FAIL=1; }

echo; echo "End: $(date)"
if [ "$FAIL" -ne 0 ]; then echo "Q3O_DECODE: BLOCKED"; exit 1; fi
echo "Q3O_DECODE: PASS"
