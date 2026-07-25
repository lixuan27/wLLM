"""Seed .wllm/traces/beta.jsonl with the six measured Beta outcomes.

Thin CLI over ``wllm.control.tracestore.seed_beta_traces`` — the seed
data lives next to the store so tests exercise the exact same function
against a tmp path. Every value is copied from docs/ALPHA_REPORT.md or
docs/BETA_REPORT.md (real SLURM jobs 195301-196293); nothing is
invented here. Idempotent: dedup by trace_id means running the script
twice adds nothing new.

Run:
    python scripts/seed_traces_beta.py [--path .wllm/traces/beta.jsonl]
Exit 0 on success (all six seed traces present afterwards).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.control.tracestore import (
    TraceStore, beta_seed_traces, seed_beta_traces,
)

DEFAULT_PATH = ".wllm/traces/beta.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_PATH)
    args = ap.parse_args()
    path = Path(args.path)
    if not path.is_absolute():
        path = ROOT / path

    store = TraceStore(path)
    if store.corrupt_lines:
        print(f"seed_traces: WARNING {store.corrupt_lines} corrupt "
              f"line(s) skipped on load (kept on disk, not rewritten)")

    expected = len(beta_seed_traces())
    added = seed_beta_traces(store)
    print(f"seed_traces: {added} added, {store.deduped} already "
          f"present (dedup), {len(store.all())} total -> {path}")
    for t in store.all():
        print(f"  {t.trace_id} {t.status:8s} {t.model} :: "
              f"{t.candidate.get('pass')} [{t.recorded}]")

    seeded = {t.trace_id for t in beta_seed_traces()}
    present = {t.trace_id for t in store.all()}
    missing = seeded - present
    if missing:
        print(f"seed_traces: {len(missing)}/{expected} seed traces "
              f"missing after seeding ({sorted(missing)}) — FAIL")
        return 1
    print(f"seed_traces: all {expected} seed traces present "
          f"(accepted={len(store.query(status='accepted'))}, "
          f"rejected={len(store.query(status='rejected'))}) — PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
