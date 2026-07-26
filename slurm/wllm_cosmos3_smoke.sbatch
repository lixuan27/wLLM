#!/bin/bash
#SBATCH --job-name=wllm_cosmos3_smoke_r4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=00:45:00
#SBATCH --exclude=server13,server59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Cosmos3-Nano onboarding: Launchable-tier smoke (fixed-seed T2I) plus
# the honest control-plane view of a not-yet-profiled model.

set -uo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"
cd /public/home/lixuan/lixuan/wllm-infra
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/public/home/lixuan/lixuan/wllm-infra/checkpoints/hf_guardrail_cache HF_HUB_OFFLINE=1
export PATH=/public/home/lixuan/lixuan/wllm-infra/.venv/bin:$PATH
TODAY=$(date +%F)
FAIL=0
step() { echo; echo "== $1"; }

step "cosmos3 0: asset readiness (content-level, fail-closed)"
.venv/bin/python - << 'PYEOF' || FAIL=1
import sys
sys.path.insert(0, ".")
from pathlib import Path
from wllm.backends.catalog.assets import (AssetSpec, ExpectedFile,
                                          check_asset)
root = "/public/home/lixuan/lixuan/pretrained-model/Cosmos3-Nano"
shards = sorted(p.relative_to(root).as_posix()
                for p in Path(root).rglob("*.safetensors"))
expected = [ExpectedFile("model_index.json", min_bytes=2,
                         must_parse_json=True)]
expected += [ExpectedFile(s, min_bytes=1024) for s in shards]
rep = check_asset(AssetSpec("cosmos3-nano", root, expected))
print(f"readiness: ready={rep.ready} checked={rep.checked}")
for b in rep.blockers:
    print(f"  BLOCKER: {b}")
raise SystemExit(0 if rep.ready and rep.checked > 1 else 1)
PYEOF

step "cosmos3 1: Launchable-tier T2I smoke (fixed seed, 20 steps)"
.venv/bin/python benchmarks/smoke_cosmos3.py \
  | tee /tmp/cosmos3_smoke_$SLURM_JOB_ID.txt || FAIL=1
grep -q "COSMOS3_SMOKE_OK" /tmp/cosmos3_smoke_$SLURM_JOB_ID.txt || FAIL=1

step "cosmos3 2: honest control-plane view (no profile yet)"
.venv/bin/python scripts/seed_traces_beta.py || FAIL=1
.venv/bin/python -m wllm.control.cli candidates . \
  --model nvidia/Cosmos3-Nano --hardware 1xH200 --num-gpus 1 \
  --traces .wllm/traces/beta.jsonl --today "$TODAY" \
  | tee /tmp/cosmos3_cand_$SLURM_JOB_ID.txt
RC=${PIPESTATUS[0]}
# diagnose-only (rc 3) is the CORRECT verdict for an unprofiled model
[ "$RC" -eq 3 ] || { echo "cosmos3: expected diagnose-only rc 3, got $RC"; FAIL=1; }
grep -q "no verified profile matches" /tmp/cosmos3_cand_$SLURM_JOB_ID.txt || FAIL=1

step "cosmos3 3: record the Launchable baseline into the trace store"
WLLM_C3_TODAY="$TODAY" WLLM_C3_JOB="$SLURM_JOB_ID" \
  .venv/bin/python - << 'PYEOF' || FAIL=1
import glob, json, os, sys
sys.path.insert(0, ".")
from wllm.control.tracestore import Trace, TraceStore
files = sorted(glob.glob("benchmarks/results/cosmos3_smoke_*.json"),
               key=os.path.getmtime)
summary = json.load(open(files[-1]))
store = TraceStore(".wllm/traces/beta.jsonl")
tid = store.append(Trace(
    model="nvidia/Cosmos3-Nano", hardware="1xH200",
    runtime="torch-local", workload="t2i-smoke 480x832, 20 steps",
    candidate={"pass": "unoptimized_baseline", "gpus": 1},
    status="accepted",
    reason="Launchable-tier baseline: model runs end-to-end here; "
           "no optimization claimed",
    metrics={"median_ms": summary["median_ms"],
             "load_s": summary["load_s"],
             "peak_vram_gb": summary["peak_vram_gb"]},
    evidence=f"job {os.environ['WLLM_C3_JOB']} ({files[-1]})",
    recorded=os.environ["WLLM_C3_TODAY"]))
print(f"cosmos3 baseline trace: id={tid} median_ms={summary['median_ms']}")
PYEOF

echo; echo "End: $(date)"
if [ "$FAIL" -ne 0 ]; then echo "COSMOS3_ONBOARD: BLOCKED"; exit 1; fi
echo "COSMOS3_ONBOARD: PASS"
