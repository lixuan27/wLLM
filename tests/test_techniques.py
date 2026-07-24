"""Technique executors: cache reuse, quant numerics, fail-closed orchestration."""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.techniques import (
    QualityBudget, QuantizedLinears, StepResidualCache,
    TechniqueOrchestrator, TechniqueSpec,
)
from wllm.techniques.orchestrator import max_rel_deviation
from wllm.techniques.step_cache import run_loop


def _smooth_step(x, k):
    """A slowly-contracting refinement step (well-suited to caching)."""
    return [v * 0.99 + 0.01 for v in x]


def _jumpy_step(x, k):
    """Alternating dynamics: inputs change a lot every step."""
    return [v * (2.0 if k % 2 == 0 else 0.4) + 1.0 for v in x]


X0 = [1.0, -2.0, 3.0, 0.5]


# --------------------------------------------------------------- step cache

def test_cache_zero_threshold_is_exact_passthrough():
    ref = run_loop(_smooth_step, X0, 20)
    cache = StepResidualCache(_smooth_step, threshold=0.0)
    got = run_loop(cache, X0, 20)
    assert got == ref
    assert cache.steps_reused == 0 and cache.steps_total == 20


def test_cache_reuses_on_smooth_trajectories_within_tolerance():
    ref = run_loop(_smooth_step, X0, 20)
    cache = StepResidualCache(_smooth_step, threshold=0.05)
    got = run_loop(cache, X0, 20)
    assert cache.steps_reused > 0                      # engaged
    assert max_rel_deviation(ref, got) < 0.05          # bounded drift
    auth = cache.authenticity()
    assert auth["steps_reused"] == float(cache.steps_reused)


def test_cache_does_not_engage_on_jumpy_trajectories():
    cache = StepResidualCache(_jumpy_step, threshold=0.05)
    ref = run_loop(_jumpy_step, X0, 12)
    got = run_loop(cache, X0, 12)
    assert cache.steps_reused == 0                     # honest: no reuse
    assert got == ref


def test_cache_consecutive_reuse_cap_forces_recompute():
    calls = []

    def counting_step(x, k):
        calls.append(k)
        return _smooth_step(x, k)

    cache = StepResidualCache(counting_step, threshold=0.9,
                              max_consecutive_reuses=3)
    run_loop(cache, X0, 13)
    # pattern: compute, then at most 3 reuses before a forced recompute
    assert cache.steps_reused > 0
    assert len(calls) >= (13 - 1) // 4 + 1
    gaps = [b - a for a, b in zip(calls, calls[1:])]
    assert max(gaps) <= 4        # never more than 3 skipped steps in a row


def test_cache_reset_clears_state_and_counters():
    cache = StepResidualCache(_smooth_step, threshold=0.05)
    run_loop(cache, X0, 10)
    assert cache.steps_reused > 0
    cache.reset()
    # stale evidence must not let a never-engaged run pass authenticity
    assert cache.steps_total == 0 and cache.steps_reused == 0
    got = run_loop(cache, X0, 10)
    fresh = StepResidualCache(_smooth_step, threshold=0.05)
    assert got == run_loop(fresh, X0, 10)   # no cross-request residue


def test_cache_guards():
    try:
        StepResidualCache(_smooth_step, threshold=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative threshold must be rejected")
    try:
        run_loop(_smooth_step, X0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero iterations must be rejected")


# ---------------------------------------------------------------- quant sim

def test_quant_numerics_bounded_and_counted():
    w = {"proj": [[0.5, -1.0, 2.0], [0.1, 0.0, -0.3]]}
    q = QuantizedLinears(w)
    assert q.layers_quantized == 1
    x = [1.0, 2.0, -1.0]
    exact = q.forward_exact("proj", x)
    approx = q.forward("proj", x)
    # int8 absmax rounding keeps per-element weight error <= scale/2
    assert q.max_weight_error() <= max(abs(v) for r in w["proj"] for v in r) / 127.0 / 2 + 1e-12
    assert max_rel_deviation(exact, approx) < 0.02
    assert q.authenticity()["layers_quantized"] == 1.0


def test_quant_guards():
    try:
        QuantizedLinears({"bad": [[1.0, 2.0], [3.0]]})
    except ValueError:
        pass
    else:
        raise AssertionError("ragged weights must be rejected")
    q = QuantizedLinears({"proj": [[1.0, 2.0]]})
    try:
        q.forward("proj", [1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("dim mismatch must be rejected")
    zero = QuantizedLinears({"z": [[0.0, 0.0]]})
    assert zero.forward("z", [5.0, 5.0]) == [0.0]


# ------------------------------------------------------------- orchestrator

def _spec(name, signals=("steps_reused",), quality="bounded"):
    return TechniqueSpec(name=name, family="cache", quality_class=quality,
                         authenticity_signals=list(signals))


def test_orchestrator_accepts_engaged_within_budget():
    ref = lambda: run_loop(_smooth_step, X0, 20)

    def cached():
        c = StepResidualCache(_smooth_step, threshold=0.05)
        out = run_loop(c, X0, 20)
        return out, c.authenticity()

    orch = TechniqueOrchestrator(ref, QualityBudget(max_rel_deviation=0.05))
    verdicts = orch.evaluate([(_spec("step_cache_t05"), cached)])
    v = verdicts[0]
    assert v.accepted and v.max_rel_deviation < 0.05
    assert v.authenticity["steps_reused"] > 0
    rec = v.receipt_fields()
    assert rec["quality"]["verdict"] == "bounded"
    assert rec["authenticity"]["steps_reused"] is True


def test_orchestrator_rejects_never_engaged():
    ref = lambda: run_loop(_jumpy_step, X0, 12)

    def cached():
        c = StepResidualCache(_jumpy_step, threshold=0.05)
        return run_loop(c, X0, 12), c.authenticity()

    orch = TechniqueOrchestrator(ref, QualityBudget(max_rel_deviation=0.5))
    v = orch.evaluate([(_spec("cache_never_engaged"), cached)])[0]
    assert not v.accepted and "never engaged" in v.reason


def test_orchestrator_rejects_budget_violation_and_crash():
    ref = lambda: run_loop(_smooth_step, X0, 20)

    def too_aggressive():
        c = StepResidualCache(_smooth_step, threshold=0.9)
        return run_loop(c, X0, 20), c.authenticity()

    def crashes():
        raise RuntimeError("kaboom")

    orch = TechniqueOrchestrator(ref, QualityBudget(max_rel_deviation=1e-6))
    verdicts = orch.evaluate([
        (_spec("cache_t90"), too_aggressive),
        (_spec("boom"), crashes),
    ])
    by = {v.spec.name: v for v in verdicts}
    assert not by["cache_t90"].accepted
    assert "quality budget exceeded" in by["cache_t90"].reason
    assert not by["boom"].accepted and "crashed" in by["boom"].reason
    assert by["boom"].receipt_fields()["quality"]["verdict"] == "failed"


def test_orchestrator_rejects_undeclared_signals_and_shape_drift():
    ref = lambda: list(X0)
    no_signals = TechniqueSpec(name="anon", family="cache",
                               authenticity_signals=[])
    def identity():
        return list(X0), {"steps_reused": 1.0}
    def wrong_shape():
        return list(X0)[:-1], {"steps_reused": 1.0}
    orch = TechniqueOrchestrator(ref, QualityBudget.exact())
    verdicts = orch.evaluate([
        (no_signals, identity),
        (_spec("short"), wrong_shape),
    ])
    by = {v.spec.name: v for v in verdicts}
    assert not by["anon"].accepted and "invalid spec" in by["anon"].reason
    assert not by["short"].accepted and "shape violation" in by["short"].reason


def test_orchestrator_refuses_empty_reference():
    orch = TechniqueOrchestrator(lambda: [], QualityBudget.exact())
    try:
        orch.evaluate([(_spec("any"), lambda: ([], {"steps_reused": 1.0}))])
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("an empty oracle must refuse to grade")


def test_orchestrator_exact_budget_accepts_exact_passthrough():
    ref = lambda: run_loop(_smooth_step, X0, 10)
    def passthrough():
        c = StepResidualCache(_smooth_step, threshold=0.0)
        out = run_loop(c, X0, 10)
        return out, {"steps_total": float(c.steps_total)}
    spec = TechniqueSpec(name="noop", family="cache", quality_class="exact",
                         authenticity_signals=["steps_total"])
    v = TechniqueOrchestrator(ref, QualityBudget.exact()).evaluate(
        [(spec, passthrough)])[0]
    assert v.accepted and v.max_rel_deviation == 0.0
    assert v.receipt_fields()["quality"]["verdict"] == "exact"


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
