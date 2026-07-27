#!/bin/bash
#SBATCH --job-name=wllm_ci_cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --exclude=server13,server59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -uo pipefail
cd /public/home/lixuan/lixuan/wllm-infra
echo "Job ID: $SLURM_JOB_ID"; echo "Node: $SLURM_NODELIST"; echo "Start: $(date)"
source .venv/bin/activate
FAIL=0
step() { echo; echo "=== $1 ==="; }

step "release gate (full tree)"
bash scripts/release_gate.sh --all || FAIL=1

step "syntax sweep (tracked python)"
git ls-files '*.py' | xargs -n 50 python -m py_compile || FAIL=1

step "unit + integration tests (pytest)"
python -m pytest tests/ -q || FAIL=1

step "BDD feature scenarios"
python tests/test_bdd_scenarios.py || FAIL=1

step "coverage gate (control plane + data-plane pillars + verifier >= 85%)"
python -m coverage run -m pytest -q tests/test_control_plane.py tests/test_bdd_scenarios.py \
    tests/test_composite.py tests/test_techniques.py tests/test_omni_engine.py \
    tests/test_catalog_assets.py tests/test_evidence.py tests/test_mcp_server.py \
    tests/test_technique_composer.py tests/test_composite_lowering.py \
    tests/test_slo.py tests/test_profiles.py tests/test_tracestore.py \
    tests/test_candidates.py tests/test_composite_walks.py \
    tests/test_guardrail.py \
    tests/test_adjudicate.py tests/test_planner_toy.py \
    tests/test_step_cache_tensor.py tests/test_plugin_package.py \
    tests/test_composite_rendezvous.py \
  && python -m coverage report --fail-under=85 | tee .wllm_coverage.txt || FAIL=1

step "real-evidence receipt (wan22 CFG branch-parallel, job 196293)"
python scripts/receipt_wan22_cfgpar.py || FAIL=1

step "optimization trace seed (idempotent, real beta outcomes)"
python scripts/seed_traces_beta.py || FAIL=1

step "measured failure prunes a candidate (three-source planner loop)"
# The reuse cache was measured at 3.30x on this exact model and refused
# on quality (job 202206).  A planner scoring on speed alone would
# promote it; ours must drop it and cite the trace.
#
# Two independent layers must hold, and the first would mask the second:
#   1. under the default exact policy the pass never even reaches
#      history, because a bounded-quality pass is illegal outright;
#   2. under a bounded-allowing policy — where the pass IS legal and
#      genuinely tempting — measured history is the thing that stops it.
# Asserting only (1) would let (2) rot silently, so both are checked.
CAND="python -m wllm.control.cli candidates . --model Wan-AI/Wan2.2-TI2V-5B \
  --hardware 1xH200 --num-gpus 1 --traces .wllm/traces/beta.jsonl \
  --today $(date +%F)"
$CAND > /tmp/wllm_cand_exact_$SLURM_JOB_ID.txt 2>&1
grep -q "reject wllm-serving:reuse_cache: quality=bounded" \
  /tmp/wllm_cand_exact_$SLURM_JOB_ID.txt \
  && echo "  layer 1 (quality policy): bounded pass refused — PASS" \
  || { echo "  layer 1: reuse_cache not refused under exact policy — FAIL"; \
       cat /tmp/wllm_cand_exact_$SLURM_JOB_ID.txt; FAIL=1; }
$CAND --quality-policy bounded > /tmp/wllm_cand_b_$SLURM_JOB_ID.txt 2>&1
# Assert the mechanism (dropped, citing a trace), not one trace's prose:
# a later measurement legitimately supersedes an earlier one by the
# store's latest-wins rule, so pinning a specific reason string would
# make this gate fail every time the evidence is refreshed — which is
# the opposite of what it is here to protect.
grep -qE "dropped pass 'reuse_cache': known-bad from trace [0-9a-f]+" \
  /tmp/wllm_cand_b_$SLURM_JOB_ID.txt \
  && echo "  layer 2 (measured history): reuse_cache dropped citing a trace — PASS" \
  || { echo "  layer 2: reuse_cache survived measured history — FAIL"; \
       cat /tmp/wllm_cand_b_$SLURM_JOB_ID.txt; FAIL=1; }

step "mutation smoke (kill rate >= 80%)"
python scripts/mutation_smoke.py --threshold 0.8 || FAIL=1

echo; echo "End: $(date)"
if [ "$FAIL" -ne 0 ]; then echo "CI: BLOCKED"; exit 1; fi
echo "CI: PASS"
