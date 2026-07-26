#!/bin/bash
#SBATCH --job-name=wllm_vjepa2_smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --exclude=server13,server59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# World-model archetype onboarding: predictor-rollout Launchable smoke
# on the locally staged checkpoint, plus the honest control-plane view.

set -uo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"
cd /public/home/lixuan/lixuan/wllm-infra
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH=/public/home/lixuan/lixuan/wllm-infra/.venv/bin:$PATH
TODAY=$(date +%F)
FAIL=0
step() { echo; echo "== $1"; }

step "vjepa2 1: Launchable predictor-rollout smoke (fixed seed)"
.venv/bin/python benchmarks/smoke_vjepa2.py \
  | tee /tmp/vjepa2_smoke_$SLURM_JOB_ID.txt || FAIL=1
grep -q "VJEPA2_SMOKE_OK" /tmp/vjepa2_smoke_$SLURM_JOB_ID.txt || FAIL=1

step "vjepa2 2: honest control-plane view (no profile yet)"
.venv/bin/python scripts/seed_traces_beta.py || FAIL=1
.venv/bin/python -m wllm.control.cli candidates . \
  --model facebook/vjepa2-vitl-fpc64-256 --hardware 1xH200 --num-gpus 1 \
  --traces .wllm/traces/beta.jsonl --today "$TODAY" \
  | tee /tmp/vjepa2_cand_$SLURM_JOB_ID.txt
RC=${PIPESTATUS[0]}
[ "$RC" -eq 3 ] || { echo "vjepa2: expected diagnose-only rc 3, got $RC"; FAIL=1; }

step "vjepa2 3: record the Launchable baseline into the trace store"
WLLM_VJ_TODAY="$TODAY" WLLM_VJ_JOB="$SLURM_JOB_ID" \
  .venv/bin/python - << 'PYEOF' || FAIL=1
import glob, json, os, sys
sys.path.insert(0, ".")
from wllm.control.tracestore import Trace, TraceStore
files = sorted(glob.glob("benchmarks/results/vjepa2_smoke_*.json"),
               key=os.path.getmtime)
summary = json.load(open(files[-1]))
store = TraceStore(".wllm/traces/beta.jsonl")
tid = store.append(Trace(
    model="facebook/vjepa2-vitl-fpc64-256", hardware="1xH200",
    runtime="torch-local",
    workload=f"predictor rollout x{summary['steps']}, "
             f"{summary['frames']}f@{summary['size']}",
    candidate={"pass": "unoptimized_baseline", "gpus": 1,
               "rollout_path": summary["rollout_path"]["mode"]},
    status="accepted",
    reason="Launchable-tier baseline: world-model predictor loop runs "
           "end-to-end here; no optimization claimed; base masked "
           "prediction, not the action-conditioned task",
    metrics={"rollout_ms": summary["rollout_ms"],
             "per_step_ms": summary["per_step_ms"],
             "encode_ms": summary["encode_ms"],
             "peak_vram_gb": summary["peak_vram_gb"]},
    evidence=f"job {os.environ['WLLM_VJ_JOB']} ({files[-1]})",
    recorded=os.environ["WLLM_VJ_TODAY"]))
print(f"vjepa2 baseline trace: id={tid} "
      f"per_step_ms={summary['per_step_ms']} "
      f"path={summary['rollout_path']['mode']}")
PYEOF

echo; echo "End: $(date)"
if [ "$FAIL" -ne 0 ]; then echo "VJEPA2_ONBOARD: BLOCKED"; exit 1; fi
echo "VJEPA2_ONBOARD: PASS"
