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

step "coverage gate (control plane >= 85%)"
python -m coverage run -m pytest -q tests/test_control_plane.py tests/test_bdd_scenarios.py \
  && python -m coverage report --fail-under=85 | tee .wllm_coverage.txt || FAIL=1

step "mutation smoke (kill rate >= 80%)"
python scripts/mutation_smoke.py --threshold 0.8 || FAIL=1

echo; echo "End: $(date)"
if [ "$FAIL" -ne 0 ]; then echo "CI: BLOCKED"; exit 1; fi
echo "CI: PASS"
