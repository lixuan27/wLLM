"""Evidence-to-receipt pipeline: real log schema, fail-closed on absence."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.control.evidence import (
    build_receipt, parse_phase_log, perf_distribution, quality_from_parity,
)
from wllm.control.registry import parse_capability

LOG = """\
Job ID: 4242
== phase 1: E2E single-GPU reference (sequential branches + decode)
[load] mode=ref1 world=1
{
 "mode": "seq",
 "median_ms": 5761.647306382656,
 "times_ms": [
  5761.647306382656,
  5788.9569294452667
 ],
 "steps": 20
}
== phase 2: E2E single-GPU batched CFG + decode
[load] mode=batched world=1
{
 "median_ms": 5666.786856949329,
 "times_ms": [
  5666.786856949329,
  5665.426203981042
 ]
}
== phase 3: E2E 2-GPU CFG branch parallel + rank0 decode
[load] mode=par2 world=2
{
 "median_ms": 4002.3984983563423,
 "times_ms": [
  4002.3984983563423,
  3998.11111
 ]
}
{"pair": "frames_ref1_vs_par2", "max_abs": 0, "mean_abs": 0.0, "bitexact": true}
{"pair": "frames_ref1_vs_batched", "max_abs": 251, "mean_abs": 5.3, "bitexact": false}
E2E_EXACT_GATE_PASS
"""

PHASE_REF = "E2E single-GPU reference (sequential branches + decode)"
PHASE_PAR2 = "E2E 2-GPU CFG branch parallel + rank0 decode"
PHASE_BATCHED = "E2E single-GPU batched CFG + decode"


def _cap(patterns=("falling back to eager",)):
    return parse_capability({
        "backend": "toy-serving", "version": "1",
        "models": {"compatible": ["*"]},
        "invariants": {"forbidden_log_patterns": list(patterns)},
    })


def test_parse_phase_log_extracts_everything():
    ev = parse_phase_log(LOG)
    assert set(ev.phases) == {PHASE_REF, PHASE_BATCHED, PHASE_PAR2}
    assert abs(ev.phases[PHASE_REF] - 5761.647306382656) < 1e-6
    assert len(ev.times_ms[PHASE_PAR2]) == 2
    assert len(ev.parity) == 2
    assert ev.parity_for("frames_ref1_vs_par2")["bitexact"] is True
    assert ev.gate_markers == ["E2E_EXACT_GATE_PASS"]


def test_perf_distribution_and_quality_helpers():
    d = perf_distribution(100.0, [90.0, 100.0, 110.0])
    assert d["p50_ms"] == 100.0 and d["p95_ms"] == 110.0 and d["samples"] == 3
    assert perf_distribution(42.0, None) == {"p50_ms": 42.0, "p95_ms": 42.0,
                                             "samples": 0}
    assert quality_from_parity({"bitexact": True, "max_abs": 0})["verdict"] \
        == "exact"
    assert quality_from_parity({"bitexact": False, "max_abs": 251})["verdict"] \
        == "failed"
    assert quality_from_parity(None)["verdict"] is None


def test_bitexact_candidate_promotes_with_real_speedup():
    ev = parse_phase_log(LOG)
    rec = build_receipt(
        "par2", _cap(), candidate_phase=PHASE_PAR2, baseline_phase=PHASE_REF,
        evidence=ev, log_text=LOG, parity_pair="frames_ref1_vs_par2",
        passes=["cfg_branch_parallel"],
        authenticity={"two_gpu_branch_execution": True},
        source_revision="abc", hardware="2xTEST")
    assert rec.promote_problems("exact") == []
    assert 1.43 < rec.speedup() < 1.45
    assert rec.authenticity["e2e_gate_marker_present"] is True


def test_non_exact_candidate_is_refused():
    ev = parse_phase_log(LOG)
    rec = build_receipt(
        "batched", _cap(), candidate_phase=PHASE_BATCHED,
        baseline_phase=PHASE_REF, evidence=ev, log_text=LOG,
        parity_pair="frames_ref1_vs_batched", passes=["cfg_batched"],
        authenticity={"batched_execution": True})
    problems = rec.promote_problems("exact")
    assert any("failed" in p for p in problems)


def test_missing_evidence_blocks_not_defaults():
    ev = parse_phase_log("no phases here\n")
    rec = build_receipt(
        "ghost", _cap(), candidate_phase=PHASE_PAR2,
        baseline_phase=PHASE_REF, evidence=ev, log_text="",
        parity_pair="frames_ref1_vs_par2", passes=["x"],
        authenticity={"ran": True}, require_gate_marker=True)
    problems = rec.promote_problems("exact")
    joined = " ".join(problems)
    assert "perf.p50_ms" in joined            # no measured distribution
    assert "quality verdict missing" in joined
    assert any("e2e_gate_marker_present" in p for p in problems)


def test_execution_markers_beat_phase_labels():
    """The funnel-job lesson: a phase label is not proof of execution.

    A log whose phase-3 chunk shows the WRONG mode (label says branch
    parallel, execution says reference) must yield phase_ran False, and
    a receipt keyed on that marker must be refused.
    """
    ev = parse_phase_log(LOG)
    assert ev.phase_ran(PHASE_PAR2, "mode=par2 world=2")
    assert ev.phase_ran(PHASE_REF, "mode=ref1 world=1")
    wrong = LOG.replace("[load] mode=par2 world=2",
                        "[load] mode=ref1 world=2")
    ev_wrong = parse_phase_log(wrong)
    assert not ev_wrong.phase_ran(PHASE_PAR2, "mode=par2 world=2")
    rec = build_receipt(
        "par2", _cap(), candidate_phase=PHASE_PAR2,
        baseline_phase=PHASE_REF, evidence=ev_wrong, log_text=wrong,
        parity_pair="frames_ref1_vs_par2", passes=["cfg_branch_parallel"],
        authenticity={"two_gpu_branch_execution":
                      ev_wrong.phase_ran(PHASE_PAR2, "mode=par2 world=2")})
    problems = rec.promote_problems("exact")
    assert any("two_gpu_branch_execution" in p for p in problems)


def test_min_speedup_blocks_no_gain_plans():
    ev = parse_phase_log(LOG)
    rec = build_receipt(
        "par2", _cap(), candidate_phase=PHASE_PAR2, baseline_phase=PHASE_REF,
        evidence=ev, log_text=LOG, parity_pair="frames_ref1_vs_par2",
        passes=["cfg_branch_parallel"], authenticity={"ran": True})
    assert rec.promote_problems("exact", min_speedup=1.1) == []   # 1.44x
    no_gain = build_receipt(
        "batched-as-candidate", _cap(), candidate_phase=PHASE_BATCHED,
        baseline_phase=PHASE_REF, evidence=ev, log_text=LOG,
        parity_pair="frames_ref1_vs_par2",   # borrow the exact verdict
        passes=["x"], authenticity={"ran": True})
    problems = no_gain.promote_problems("exact", min_speedup=1.1)
    assert any("no effective optimization" in p for p in problems)


def test_forbidden_log_pattern_invalidates():
    ev = parse_phase_log(LOG)
    dirty = LOG + "\nwarn: Falling Back To Eager mode\n"
    rec = build_receipt(
        "par2", _cap(), candidate_phase=PHASE_PAR2, baseline_phase=PHASE_REF,
        evidence=ev, log_text=dirty, parity_pair="frames_ref1_vs_par2",
        passes=["cfg_branch_parallel"], authenticity={"ran": True})
    assert rec.fallback_hits
    assert any("silent fallback" in p for p in rec.promote_problems("exact"))


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {str(exc)[:200]}")
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
