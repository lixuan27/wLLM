"""SLO compiler: hard constraints, lifecycle amortization, weighted choice."""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.control.cli import main as cli_main
from wllm.control.slo import (
    CandidateMetrics, Lifecycle, SLOSpec, admit, amortized_seconds, choose,
    pareto_profiles,
)


def _slo(**over) -> SLOSpec:
    d = {"hard_constraints": {}, "preferences": {}, "lifecycle": {}}
    d.update(over)
    return SLOSpec.from_dict(d)


# ------------------------------------------------------------------ schema

def test_spec_validation_fail_closed():
    assert SLOSpec().validate() == []
    s = _slo(hard_constraints={"vibes": 1, "p99_latency_ms": -5,
                               "api_compatibility": "sorta"},
             preferences={"latency": 0.4, "speed": 0.6})
    errs = " ".join(s.validate())
    for frag in ("vibes", "p99_latency_ms", "api_compatibility", "speed"):
        assert frag in errs
    lop = _slo(preferences={"latency": 0.9, "cost": 0.3})
    assert any("sum to 1.0" in e for e in lop.validate())
    ok = _slo(preferences={"latency": 0.6, "throughput": 0.4})
    assert ok.validate() == []
    try:
        SLOSpec.from_dict({"turbo": True})
    except ValueError as exc:
        assert "turbo" in str(exc)
    else:
        raise AssertionError("unknown SLO fields must be rejected")


def test_amortized_math():
    # (60 + 100*1 + 0) / 100 = 1.6 s per request
    assert abs(amortized_seconds(60.0, 1.0, 100) - 1.6) < 1e-9
    try:
        amortized_seconds(1.0, 1.0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("n_requests=0 must be rejected")


# ------------------------------------------------------------------- admit

def test_admit_unmeasured_is_violation():
    s = _slo(hard_constraints={"p99_latency_ms": 2000, "peak_vram_gb": 80})
    ok = CandidateMetrics("a", p50_ms=500, p99_ms=1500, peak_vram_gb=60)
    assert admit(s, ok) == []
    over = CandidateMetrics("b", p50_ms=500, p99_ms=2500, peak_vram_gb=60)
    assert any("exceeds limit" in v for v in admit(s, over))
    unmeasured = CandidateMetrics("c", p50_ms=500, p99_ms=1500)
    assert any("no peak_vram_gb measurement" in v
               for v in admit(s, unmeasured))


# ------------------------------------------------------------------ pareto

def test_pareto_labels():
    cands = [
        CandidateMetrics("fast", p50_ms=100, throughput_rps=5,
                         startup_s=60, cost_per_request=0.02, exact=False),
        CandidateMetrics("cheap", p50_ms=300, throughput_rps=8,
                         startup_s=2, cost_per_request=0.01),
    ]
    labels = pareto_profiles(cands)
    assert labels["lowest_latency"] == "fast"
    assert labels["highest_throughput"] == "cheap"
    assert labels["lowest_startup"] == "cheap"
    assert labels["lowest_cost"] == "cheap"
    assert labels["strict_exact"] == "cheap"     # only exact candidate


# ------------------------------------------------------------------ choose

def _compile_vs_eager():
    compiled = CandidateMetrics("compiled", p50_ms=800, startup_s=60)
    eager = CandidateMetrics("eager", p50_ms=1000, startup_s=1)
    return compiled, eager


def test_lifecycle_flips_the_choice():
    """The master insight: startup cost amortizes over replica lifetime."""
    compiled, eager = _compile_vs_eager()
    long_lived = _slo(preferences={"latency": 1.0},
                      lifecycle={"expected_requests_per_replica": 10000})
    short_lived = _slo(preferences={"latency": 1.0},
                       lifecycle={"expected_requests_per_replica": 20})
    assert choose(long_lived, [compiled, eager]).chosen == "compiled"
    # 20 requests: compiled amortizes to 3800 ms/req, eager to 1050
    assert choose(short_lived, [compiled, eager]).chosen == "eager"


def test_choose_hard_constraints_and_missing_metrics():
    s = _slo(hard_constraints={"p99_latency_ms": 1000},
             preferences={"latency": 0.5, "throughput": 0.5})
    a = CandidateMetrics("a", p50_ms=400, p99_ms=900, throughput_rps=4)
    b = CandidateMetrics("b", p50_ms=300, p99_ms=1500, throughput_rps=9)
    c = CandidateMetrics("c", p50_ms=500, p99_ms=800)   # no throughput
    sel = choose(s, [a, b, c])
    assert "b" in sel.rejected                     # hard p99 violation
    assert sel.chosen == "a"                       # c scored worst-case tput
    assert any("no throughput measurement" in n for n in sel.notes)
    empty = choose(_slo(hard_constraints={"p99_latency_ms": 10}),
                   [a, b, c])
    assert empty.chosen is None and empty.rejected and empty.notes


def test_choose_guards():
    try:
        choose(_slo(preferences={"latency": 0.5}), [])
    except ValueError:
        pass
    else:
        raise AssertionError("empty candidates must be rejected")
    bad = _slo(preferences={"latency": 0.5, "cost": 0.2})
    try:
        choose(bad, [CandidateMetrics("x", p50_ms=1)])
    except ValueError as exc:
        assert "invalid SLO" in str(exc)
    else:
        raise AssertionError("invalid SLO must be rejected")


# --------------------------------------------------------------------- cli

def test_cli_select_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        metrics = Path(td) / "m.json"
        metrics.write_text(json.dumps([
            {"name": "compiled", "p50_ms": 800, "startup_s": 60,
             "p99_ms": 900},
            {"name": "eager", "p50_ms": 1000, "startup_s": 1,
             "p99_ms": 1100},
        ]))
        slo = Path(td) / "slo.yaml"
        slo.write_text(
            "hard_constraints:\n  p99_latency_ms: 2000\n"
            "preferences:\n  latency: 1.0\n"
            "lifecycle:\n  expected_requests_per_replica: 20\n")
        assert cli_main(["select", "--metrics", str(metrics),
                         "--slo", str(slo)]) == 0
        strict = Path(td) / "strict.yaml"
        strict.write_text("hard_constraints:\n  p99_latency_ms: 10\n")
        assert cli_main(["select", "--metrics", str(metrics),
                         "--slo", str(strict)]) == 4


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
