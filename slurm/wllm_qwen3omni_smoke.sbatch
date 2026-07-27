#!/bin/bash
#SBATCH --job-name=wllm_qwen3omni_smoke_retry1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --exclude=server13,server59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Omni thinker-talker archetype onboarding: thinker-only text smoke.

set -uo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"
cd /public/home/lixuan/lixuan/wllm-infra
export WLLM_Q3O_PATH=/public/home/lixuan/lixuan/pretrained-model/Qwen3-Omni-30B-A3B-Instruct
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH=/public/home/lixuan/lixuan/wllm-infra/.venv/bin:$PATH
TODAY=$(date +%F)
FAIL=0
step() { echo; echo "== $1"; }

step "qwen3omni 1: thinker text smoke (fixed prompt, greedy)"
.venv/bin/python benchmarks/smoke_qwen3omni.py \
  | tee /tmp/q3o_smoke_$SLURM_JOB_ID.txt || FAIL=1
grep -q "QWEN3OMNI_SMOKE_OK" /tmp/q3o_smoke_$SLURM_JOB_ID.txt || FAIL=1

step "qwen3omni 2: honest control-plane view (registry-only knowledge)"
# retry1 root cause: this model hits the registry's Qwen/* compatible
# glob, so the planner legitimately emits registry-only candidates
# (rc 0) with the no-verified-profile caveat; rc 3 diagnose-only is
# reserved for models no backend claims at all.
.venv/bin/python scripts/seed_traces_beta.py || FAIL=1
.venv/bin/python -m wllm.control.cli candidates . \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct --hardware 1xH200 --num-gpus 1 \
  --traces .wllm/traces/beta.jsonl --today "$TODAY" \
  | tee /tmp/q3o_cand_$SLURM_JOB_ID.txt
RC=$?
[ "$RC" -eq 0 ] || { echo "qwen3omni: candidates rc $RC != 0"; FAIL=1; }
grep -q "no verified profile matches this model" \
  /tmp/q3o_cand_$SLURM_JOB_ID.txt \
  || { echo "qwen3omni: missing no-profile caveat"; FAIL=1; }
grep -q "candidate wllm-serving \[compatible\]" \
  /tmp/q3o_cand_$SLURM_JOB_ID.txt \
  || { echo "qwen3omni: expected compatible-tier candidate"; FAIL=1; }

step "qwen3omni 3: record the Launchable baseline into the trace store"
WLLM_Q3O_TODAY="$TODAY" WLLM_Q3O_JOB="$SLURM_JOB_ID" \
  .venv/bin/python - << 'PYEOF' || FAIL=1
import glob, json, os, sys
sys.path.insert(0, ".")
from wllm.control.tracestore import Trace, TraceStore
files = sorted(glob.glob("benchmarks/results/qwen3omni_smoke_*.json"),
               key=os.path.getmtime)
summary = json.load(open(files[-1]))
store = TraceStore(".wllm/traces/beta.jsonl")
tid = store.append(Trace(
    model=summary["model"], hardware="1xH200", runtime="torch-local",
    workload=f"thinker text, {summary['new_tokens']} new tokens",
    candidate={"pass": "unoptimized_baseline", "gpus": 1},
    status="accepted",
    reason="Launchable-tier baseline: omni thinker text path runs "
           "end-to-end here; talker/audio chain is a later tier",
    metrics={"median_ms": summary["median_ms"],
             "tok_per_s": summary["tok_per_s"],
             "load_s": summary["load_s"],
             "peak_vram_gb": summary["peak_vram_gb"]},
    evidence=f"job {os.environ['WLLM_Q3O_JOB']} ({files[-1]})",
    recorded=os.environ["WLLM_Q3O_TODAY"]))
print(f"qwen3omni baseline trace: id={tid} "
      f"tok_per_s={summary['tok_per_s']}")
PYEOF

echo; echo "End: $(date)"
if [ "$FAIL" -ne 0 ]; then echo "QWEN3OMNI_ONBOARD: BLOCKED"; exit 1; fi
echo "QWEN3OMNI_ONBOARD: PASS"
