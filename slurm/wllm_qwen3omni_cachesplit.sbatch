#!/bin/bash
#SBATCH --job-name=wllm_qwen3omni_cachesplit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --exclude=server13,server59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Separate the static KV cache from the auto-compile that rides along
# with it on this runtime. New experiment, not a resubmission: job 202214
# measured the two together and the refusal could not be attributed.
#
# PASS covers all three publishable outcomes (cache alone exact+fast,
# exact+slow, or still inexact) AND the outcome where suppression itself
# turns out to be impossible on this build -- that last one is a finding,
# not a failure, and is reported via the Q3O_SPLIT_UNSUPPRESSED marker.
# BLOCKED means the experiment did not produce a judgeable measurement.

set -uo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"
cd /public/home/lixuan/lixuan/wllm-infra
export WLLM_Q3O_PATH=/public/home/lixuan/lixuan/pretrained-model/Qwen3-Omni-30B-A3B-Instruct
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH=/public/home/lixuan/lixuan/wllm-infra/.venv/bin:$PATH
# 50 min of experiment under a 60 min wall: a slow compile degrades into
# an honest "wall budget exceeded" row instead of an unreported TIMEOUT.
export WLLM_Q3O_BUDGET_S=3000
TODAY=$(date +%F)
FAIL=0
step() { echo; echo "== $1"; }

step "q3o-split 1: preflight (weights present, no re-download)"
if [ ! -f "$WLLM_Q3O_PATH/model.safetensors.index.json" ]; then
  echo "q3o-split: checkpoint index missing at $WLLM_Q3O_PATH"; FAIL=1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

step "q3o-split 2: cache-vs-compile split (single load, 3 legs, exact gate)"
.venv/bin/python benchmarks/cachesplit_qwen3omni.py \
  | tee /tmp/q3o_split_$SLURM_JOB_ID.txt
grep -q "Q3O_SPLIT_OK" /tmp/q3o_split_$SLURM_JOB_ID.txt \
  || { echo "q3o-split: experiment did not reach its OK marker"; FAIL=1; }

step "q3o-split 3: record every leg into the trace store"
if [ "$FAIL" -eq 0 ]; then
  WLLM_Q3O_TODAY="$TODAY" WLLM_Q3O_JOB="$SLURM_JOB_ID" \
    .venv/bin/python - << 'PYEOF' || FAIL=1
import glob, json, os, sys
sys.path.insert(0, ".")
from wllm.control.tracestore import Trace, TraceStore

files = sorted(glob.glob("benchmarks/results/qwen3omni_cachesplit_*.json"),
               key=os.path.getmtime)
if not files:
    raise SystemExit("no split summary written; nothing to record")
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
    print(f"  {tr['status']:<9} {tr['candidate']['pass']:<26} "
          f"n={tr['candidate']['new_tokens']:<4} id={tid}")
print(f"q3o-split traces: {counts} (deduped {store.deduped})")
print(f"q3o-split suppression_worked={summary['suppression_worked']} "
      f"any_exact_win={summary['any_exact_win']}")
print("q3o-split attribution:", json.dumps(summary["attribution"]))
PYEOF
else
  echo "q3o-split: skipping trace append because the experiment did not pass"
fi

step "q3o-split 4: profile pack still validates and is fresh"
.venv/bin/python tests/test_profiles.py | tail -n 3 \
  | tee /tmp/q3o_split_prof_$SLURM_JOB_ID.txt
grep -q "ALL PASS" /tmp/q3o_split_prof_$SLURM_JOB_ID.txt \
  || { echo "q3o-split: profile pack test not green"; FAIL=1; }

echo; echo "End: $(date)"
if [ "$FAIL" -ne 0 ]; then echo "Q3O_SPLIT: BLOCKED"; exit 1; fi
echo "Q3O_SPLIT: PASS"
