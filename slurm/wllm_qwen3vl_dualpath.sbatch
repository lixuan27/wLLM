#!/bin/bash
#SBATCH --job-name=wllm_qwen3vl_dualpath
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:45:00
#SBATCH --exclude=server13,server59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Integrity re-verification: does the published static-KV claim survive
# the strengthened verifier law (epsilon-optimal set + prefill/decode
# consistency)?  A claim that only stood because the gate was weaker
# must be withdrawn, so this job is allowed to fail the old result.

set -uo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"
cd /public/home/lixuan/lixuan/wllm-infra
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_CACHE_DIR=/tmp/wllm_inductor_$SLURM_JOB_ID
TODAY=$(date +%F)
FAIL=0

echo; echo "== re-run the qwen3-vl sweep under the dual-path adjudicator =="
.venv/bin/python benchmarks/baseline_qwen3vl.py \
  | tee /tmp/q3vl_dual_$SLURM_JOB_ID.txt || FAIL=1
grep -q "QWEN3VL_OK" /tmp/q3vl_dual_$SLURM_JOB_ID.txt || FAIL=1

echo; echo "== adjudication evidence actually exercised? =="
# The whole point of the re-run is the decode path.  If every verdict
# came from prefill alone, the stronger rule was never applied and this
# job proves nothing — say so instead of banking a false reassurance.
if grep -qi "consistency rule was not exercised" /tmp/q3vl_dual_$SLURM_JOB_ID.txt; then
  echo "NOTE: at least one verdict fell back to prefill-only evidence"
  grep -i "consistency rule was not exercised" /tmp/q3vl_dual_$SLURM_JOB_ID.txt | head -3
fi
grep -Ei "undecidable|benign_tie|real_divergence|identical" \
  /tmp/q3vl_dual_$SLURM_JOB_ID.txt | head -10

echo; echo "== record the re-verified verdicts (accepted AND withdrawn) =="
WLLM_Q3VL_TODAY="$TODAY" WLLM_Q3VL_JOB="$SLURM_JOB_ID" \
  .venv/bin/python - << 'PYEOF' || FAIL=1
import glob, json, os, sys
sys.path.insert(0, ".")
from wllm.control.tracestore import Trace, TraceStore
files = sorted(glob.glob("benchmarks/results/qwen3vl_opt_*.json"),
               key=os.path.getmtime)
if not files:
    print("no qwen3vl result json produced"); raise SystemExit(1)
res = json.load(open(files[-1]))
legs = res.get("legs", res if isinstance(res, list) else [])
store = TraceStore(".wllm/traces/beta.jsonl")
for leg in legs:
    name = leg.get("leg") or leg.get("name") or "unknown"
    if name in ("reference", "baseline"):
        continue
    ok = leg.get("status") in ("accepted", "ok", "pass")
    reason = leg.get("reason") or leg.get("verdict") or "no reason recorded"
    tid = store.append(Trace(
        model="Qwen/Qwen3-VL-8B-Instruct", hardware="1xH200",
        runtime="torch-local", workload=res.get("workload", "AR decode"),
        candidate={"pass": name, "gpus": 1},
        status="accepted" if ok else "rejected",
        reason=f"dual-path re-verification: {reason}",
        metrics={k: leg[k] for k in ("median_ms", "speedup", "tok_per_s")
                 if k in leg},
        evidence=f"job {os.environ['WLLM_Q3VL_JOB']} ({files[-1]})",
        recorded=os.environ["WLLM_Q3VL_TODAY"]))
    print(f"  {'accepted' if ok else 'rejected'} {name} id={tid} :: {reason}")
PYEOF

echo; echo "End: $(date)"
if [ "$FAIL" -ne 0 ]; then echo "QWEN3VL_DUALPATH: BLOCKED"; exit 1; fi
echo "QWEN3VL_DUALPATH: PASS"
