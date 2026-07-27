#!/bin/bash
#SBATCH --job-name=wllm_stepcache_wan22_v2
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
# Reuse-cache technique executor, measured on a real 5B diffusion loop.

set -uo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"
cd /public/home/lixuan/lixuan/wllm-infra
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
TODAY=$(date +%F)
FAIL=0

echo; echo "== stepcache 1: reference vs cached legs (one model load) =="
.venv/bin/python benchmarks/stepcache_wan22.py \
  | tee /tmp/stepcache_$SLURM_JOB_ID.txt || FAIL=1
grep -q "STEPCACHE_WAN22_OK" /tmp/stepcache_$SLURM_JOB_ID.txt || FAIL=1

echo; echo "== stepcache 2: write every leg (accepted AND rejected) to traces =="
WLLM_SC_TODAY="$TODAY" WLLM_SC_JOB="$SLURM_JOB_ID" \
  .venv/bin/python - << 'PYEOF' || FAIL=1
import glob, json, os, sys
sys.path.insert(0, ".")
from wllm.control.tracestore import Trace, TraceStore
files = sorted(glob.glob("benchmarks/results/wan22_stepcache_*.json"),
               key=os.path.getmtime)
summary = json.load(open(files[-1]))
store = TraceStore(".wllm/traces/beta.jsonl")
written = 0
for leg in summary["legs"]:
    if leg["status"] == "reference":
        continue
    status = {"accepted": "accepted", "rejected": "rejected",
              "failed": "rejected"}[leg["status"]]
    metrics = {k: leg[k] for k in
               ("median_ms", "speedup", "steps_reused", "steps_total",
                "psnr_db", "max_abs_255") if k in leg}
    tid = store.append(Trace(
        model=summary["model"], hardware="1xH200", runtime="torch-local",
        workload=summary["workload"],
        candidate={"pass": "reuse_cache", "site": summary["cache_site"],
                   "threshold": leg["threshold"], "key": leg.get("key", "input"),
                   "consecutive_cap": leg.get("consecutive_cap"), "gpus": 1},
        status=status, reason=leg["reason"], metrics=metrics,
        evidence=f"job {os.environ['WLLM_SC_JOB']} ({files[-1]})",
        recorded=os.environ["WLLM_SC_TODAY"]))
    written += 1
    print(f"  {status:8s} {leg['leg']} id={tid} :: {leg['reason']}")
print(f"stepcache traces written: {written}")
PYEOF

echo; echo "End: $(date)"
if [ "$FAIL" -ne 0 ]; then echo "STEPCACHE_WAN22: BLOCKED"; exit 1; fi
echo "STEPCACHE_WAN22: PASS"
