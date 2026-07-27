#!/bin/bash
#SBATCH --job-name=wllm_cosmos3_guardrail_retry1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=00:50:00
#SBATCH --exclude=server13,server59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Cosmos3-Nano safety chain restoration: measure the guardrail, then
# record the guardrail-enabled configuration as the honest baseline.
# The onboarding smoke's BLOCKER path is untouched; this job is the
# other arm, and it fails closed if the weights are not loadable.
#
# retry1 root cause (job 202227, FAILED 00:02:49): the guardrail repo's
# refs/main had been written with a trailing newline (41 bytes, not
# 40). huggingface_hub resolves an offline revision with a verbatim
# f.read() and no strip, so snapshot_download looked for
# snapshots/<sha>\n and raised LocalEntryNotFoundError inside
# CosmosSafetyChecker() -- after a 28s pipeline load and a 56s warmup.
# The readiness precheck stripped the ref and wrongly passed.
# Changes since 202227: (a) the ref is rewritten to the canonical 40
# bytes; (b) _resolve_revision now mirrors the loader byte-for-byte
# (verbatim ref, no lone-snapshot fallback) and names the offending
# repo/revision; (c) requirements cover every path
# CosmosSafetyChecker.__init__ resolves, including the separate
# Qwen3Guard repo; (d) the benchmark prints a full traceback on the
# failure path instead of a bare one-line marker.

set -uo pipefail
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"
cd /public/home/lixuan/lixuan/wllm-infra
export HF_HOME=/public/home/lixuan/lixuan/wllm-infra/checkpoints/hf_guardrail_cache
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PATH=/public/home/lixuan/lixuan/wllm-infra/.venv/bin:$PATH
TODAY=$(date +%F)
FAIL=0
step() { echo; echo "== $1"; }

step "guardrail 0: cache-tree readiness (content-level, fail-closed)"
# The loaders resolve weights through snapshot_download and walk
# parents[2]/blobs, so the HF *cache layout* is part of the asset.
.venv/bin/python - << 'PYEOF' || FAIL=1
import sys
sys.path.insert(0, ".")
from benchmarks.guardrail_cosmos3 import CACHE_ROOT, REQUIREMENTS
from wllm.control.guardrail import check_hf_cache
rep = check_hf_cache(CACHE_ROOT, REQUIREMENTS)
print(f"readiness: ready={rep.ready} checked={rep.checked}")
for repo, path in sorted(rep.resolved.items()):
    print(f"  resolved {repo} -> {path}")
for b in rep.blockers:
    print(f"  BLOCKER: {b}")
raise SystemExit(0 if rep.ready else 1)
PYEOF

if [ "$FAIL" -ne 0 ]; then
  echo; echo "COSMOS3_GUARDRAIL: BLOCKED (weights not loadable offline)"
  echo "End: $(date)"; exit 1
fi

step "guardrail 1: fixed-seed A/B (safety chain off vs on)"
.venv/bin/python benchmarks/guardrail_cosmos3.py \
  | tee /tmp/c3guard_$SLURM_JOB_ID.txt
RC=${PIPESTATUS[0]}
[ "$RC" -eq 0 ] || { echo "guardrail: benchmark rc $RC != 0"; FAIL=1; }
grep -q "COSMOS3_GUARDRAIL_OK" /tmp/c3guard_$SLURM_JOB_ID.txt \
  || { echo "guardrail: missing OK marker"; FAIL=1; }
# The benign control prompt MUST pass; a chain that refuses it is not a
# serving default no matter what its latency looks like.
grep -q "benign prompt passed the text guardrail: True" \
  /tmp/c3guard_$SLURM_JOB_ID.txt \
  || { echo "guardrail: benign control prompt did not pass"; FAIL=1; }

if [ "$FAIL" -ne 0 ]; then
  echo; echo "COSMOS3_GUARDRAIL: BLOCKED"
  echo "End: $(date)"; exit 1
fi

step "guardrail 2: record the guarded configuration as the baseline"
WLLM_C3G_TODAY="$TODAY" WLLM_C3G_JOB="$SLURM_JOB_ID" \
  .venv/bin/python - << 'PYEOF' || FAIL=1
import glob, json, os, sys
sys.path.insert(0, ".")
from wllm.control.tracestore import Trace, TraceStore
files = sorted(glob.glob("benchmarks/results/cosmos3_guardrail_*.json"),
               key=os.path.getmtime)
s = json.load(open(files[-1]))
acct = s["accounting"]
reason = (
    "Safety chain restored and measured: the guardrail-enabled "
    "configuration is the honest serving baseline for this model, and "
    "the earlier safety-off number is a lower bound that no deployment "
    "should quote. Same prompt/seed/steps, one process, interleaved "
    f"arms; benign control prompt passed ({s['benign_prompt_passed']}). "
    f"Guardrail costs +{acct['overhead_ms']:.0f} ms "
    f"({acct['overhead_pct']:.1f}%) per request on top of "
    f"{acct['baseline_ms']:.0f} ms unguarded.")
if s.get("hypothesis"):
    reason += " Follow-up, NOT claimed: " + s["hypothesis"]
store = TraceStore(".wllm/traces/beta.jsonl")
tid = store.append(Trace(
    model=s["model"], hardware="1xH200", runtime="torch-local",
    workload=s["workload"] + " [safety chain on]",
    candidate={"pass": "safety_chain_enabled", "gpus": 1,
               "guardrail": "text+frame"},
    status="accepted", reason=reason,
    metrics={"guarded_ms": round(acct["guarded_ms"], 1),
             "unguarded_ms": round(acct["baseline_ms"], 1),
             "guardrail_overhead_ms": round(acct["overhead_ms"], 1),
             "guardrail_overhead_pct": round(acct["overhead_pct"], 1),
             "text_stage_ms": round(acct["text_stage_ms"], 1),
             "video_stage_ms": round(acct["video_stage_ms"], 1),
             "guardrail_load_s": s["guardrail_load_s"],
             "peak_vram_gb": s["peak_vram_gb"]},
    evidence=f"job {os.environ['WLLM_C3G_JOB']} ({files[-1]})",
    recorded=os.environ["WLLM_C3G_TODAY"]))
print(f"cosmos3 guardrail trace: id={tid} "
      f"guarded_ms={acct['guarded_ms']:.0f} "
      f"overhead={acct['overhead_pct']:.1f}%")
PYEOF

step "guardrail 3: profile still validates and is not stale"
WLLM_C3G_TODAY="$TODAY" .venv/bin/python - << 'PYEOF' || FAIL=1
import os, sys
sys.path.insert(0, ".")
from wllm.profiles import load_profiles, match
profiles = load_profiles()
p = match(profiles, "nvidia/Cosmos3-Nano")
assert p is not None, "cosmos3 profile no longer matches its model id"
today = os.environ["WLLM_C3G_TODAY"]
assert not p.is_stale(today), f"cosmos3 profile is stale as of {today}"
print(f"profile ok: {p.model_family} "
      f"evidence_level={p.binding.evidence_level} "
      f"last_validated={p.binding.last_validated}")
PYEOF

echo; echo "End: $(date)"
if [ "$FAIL" -ne 0 ]; then echo "COSMOS3_GUARDRAIL: BLOCKED"; exit 1; fi
echo "COSMOS3_GUARDRAIL: PASS"
