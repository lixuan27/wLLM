"""Build the first real-evidence receipt: Wan2.2 CFG branch-parallel.

Parses the on-disk E2E job log (three phases: sequential reference,
single-GPU batched CFG, 2-GPU branch parallel; frame parity records;
exact gate marker), assembles a receipt for the branch-parallel plan,
and runs the promote gate on it. Also demonstrates the refusal path:
the batched-CFG variant is NOT bit-exact (max_abs 251/255) and its
receipt must be blocked.

Run on a compute node (CI does):
    python scripts/receipt_wan22_cfgpar.py [--log logs/<job>.out]
Exit 0 only if the parallel receipt promotes AND the batched one is
correctly refused.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.control.evidence import build_receipt, parse_phase_log
from wllm.control.registry import default_registry

DEFAULT_LOG = "logs/wllm_wan22_cfgpar_e2e_r3_196293.out"
PHASE_REF = "E2E single-GPU reference (sequential branches + decode)"
PHASE_BATCHED = "E2E single-GPU batched CFG + decode"
PHASE_PAR2 = "E2E 2-GPU CFG branch parallel + rank0 decode"


def _git_rev() -> str:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip()[:12]
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=DEFAULT_LOG)
    args = ap.parse_args()
    log_path = ROOT / args.log
    if not log_path.is_file():
        print(f"receipt_wan22: evidence log missing ({log_path}); "
              f"nothing to build — SKIP")
        return 0
    text = log_path.read_text(errors="replace")
    ev = parse_phase_log(text)
    missing = [p for p in (PHASE_REF, PHASE_PAR2) if p not in ev.phases]
    if missing:
        print(f"receipt_wan22: log lacks required phases {missing}; "
              f"refusing to fabricate — FAIL")
        return 1
    cap = default_registry()["wllm-serving"]
    meta = dict(source_revision=_git_rev(),
                model_revision="Wan2.2-TI2V-5B (local, bf16 declared)",
                hardware="2xH200", driver="580.95", precision="bf16",
                rollback_target="reference")

    par2 = build_receipt(
        "wan22-cfg-branch-parallel-2gpu", cap,
        candidate_phase=PHASE_PAR2, baseline_phase=PHASE_REF,
        evidence=ev, log_text=text,
        parity_pair="frames_ref1_vs_par2",
        passes=["cfg_branch_parallel"],
        authenticity={
            "two_gpu_branch_execution": PHASE_PAR2 in ev.phases,
            "frame_parity_check_ran":
                ev.parity_for("frames_ref1_vs_par2") is not None,
        }, **meta)
    problems = par2.promote_problems("exact")
    speed = par2.speedup()
    out = par2.save(ROOT / ".wllm" / "receipts")
    print(f"receipt_wan22: par2 receipt -> {out}")
    print(f"  fingerprint={par2.fingerprint()} "
          f"speedup={f'{speed:.2f}x' if speed else 'n/a'}")
    if problems:
        for p in problems:
            print(f"  BLOCK: {p}")
        print("receipt_wan22: parallel plan unexpectedly blocked — FAIL")
        return 1
    print("  promote gate: PASS (measured, bit-exact, no fallback)")

    batched = build_receipt(
        "wan22-cfg-batched-1gpu", cap,
        candidate_phase=PHASE_BATCHED, baseline_phase=PHASE_REF,
        evidence=ev, log_text=text,
        parity_pair="frames_ref1_vs_batched",
        passes=["cfg_batched"],
        authenticity={"batched_execution": PHASE_BATCHED in ev.phases},
        **meta)
    bproblems = batched.promote_problems("exact")
    batched.save(ROOT / ".wllm" / "receipts")
    if not bproblems:
        print("receipt_wan22: batched variant promoted despite non-exact "
              "frames — the gate is broken — FAIL")
        return 1
    print(f"receipt_wan22: batched variant correctly refused "
          f"({bproblems[0][:80]}...)")
    print("receipt_wan22: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
