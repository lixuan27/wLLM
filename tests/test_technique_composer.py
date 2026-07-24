"""Technique composer: composition legality is measured, not assumed."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.techniques import (
    Composable, QualityBudget, StepResidualCache, TechniqueComposer,
    TechniqueSpec,
)


def _smooth_step(x, k):
    """A slowly-contracting refinement step (well-suited to caching)."""
    return [v * 0.99 + 0.01 for v in x]


def _jumpy_step(x, k):
    """Alternating dynamics: inputs change a lot every step."""
    return [v * (2.0 if k % 2 == 0 else 0.4) + 1.0 for v in x]


X0 = [1.0, -2.0, 3.0, 0.5]


def _cache_composable(name, threshold):
    spec = TechniqueSpec(name=name, family="cache",
                         quality_class="bounded",
                         authenticity_signals=["steps_reused"])

    def make(inner):
        cache = StepResidualCache(inner, threshold=threshold)
        return cache, cache.authenticity

    return Composable(spec=spec, make=make)


def _counter_composable(name):
    """Exact no-op instrumentation: counts calls, changes nothing."""
    spec = TechniqueSpec(name=name, family="scheduling",
                         quality_class="exact",
                         authenticity_signals=["calls"])

    def make(inner):
        counts = {"calls": 0.0}

        def wrapped(x, k):
            counts["calls"] += 1.0
            return inner(x, k)

        return wrapped, lambda: dict(counts)

    return Composable(spec=spec, make=make)


# ---------------------------------------------------------- compatible pair

def test_compatible_pair_composes_and_chooses_fastest():
    composer = TechniqueComposer(
        _smooth_step, X0, 20, QualityBudget(max_rel_deviation=0.05))
    cache = _cache_composable("cache_smooth", 0.05)
    counter = _counter_composable("counter")
    report = composer.select([cache, counter])

    singles = {v.spec.name: v for v in report.singles}
    assert singles["cache_smooth"].accepted, singles["cache_smooth"].reason
    assert singles["counter"].accepted, singles["counter"].reason
    assert singles["counter"].max_rel_deviation == 0.0
    # purity pin: orchestrator repeats the runner 3 times; a shared
    # wrapper instance would report 60 calls, a fresh one exactly 20
    assert singles["counter"].authenticity["calls"] == 20.0

    combo = {v.spec.name: v for v in report.combos}["cache_smooth+counter"]
    assert combo.accepted, combo.reason
    assert combo.spec.family == "scheduling"      # mixed families
    assert combo.spec.quality_class == "bounded"  # worst of members
    # prefixed authenticity: both members engaged inside the combo,
    # and the counter (outermost, listed last) saw every loop step
    assert combo.authenticity["cache_smooth.steps_reused"] > 0
    assert combo.authenticity["cache_smooth.steps_total"] == 20.0
    assert combo.authenticity["counter.calls"] == 20.0

    accepted = [v for v in report.singles + report.combos if v.accepted]
    assert report.chosen is not None and report.chosen.accepted
    assert report.chosen.wall_ms == min(v.wall_ms for v in accepted)


# --------------------------------------------------------- interfering pair

def test_stacked_caches_interfere_and_fall_back_to_best_single():
    # each cache alone drifts ~0.007 on this trajectory; stacked, the
    # inner cache reuses a stale residual whenever the outer recomputes
    # and drift compounds to ~0.037 — superadditive, over the budget
    composer = TechniqueComposer(
        _smooth_step, X0, 20, QualityBudget(max_rel_deviation=0.03))
    a = _cache_composable("cache_a", 0.06)
    b = _cache_composable("cache_b", 0.06)
    report = composer.select([a, b])

    singles = {v.spec.name: v for v in report.singles}
    assert singles["cache_a"].accepted and singles["cache_b"].accepted
    assert len(report.combos) == 1
    combo = report.combos[0]
    assert combo.spec.name == "cache_a+cache_b"
    assert combo.spec.family == "cache"           # uniform family
    assert not combo.accepted
    assert "budget" in combo.reason
    # both members engaged — the rejection is interference, not
    # non-engagement — and the drift genuinely exceeds the member sum
    assert combo.authenticity["cache_a.steps_reused"] > 0
    assert combo.authenticity["cache_b.steps_reused"] > 0
    member_sum = (singles["cache_a"].max_rel_deviation
                  + singles["cache_b"].max_rel_deviation)
    assert combo.max_rel_deviation > member_sum

    assert any("interference" in msg for msg in report.interference)
    assert report.chosen is not None
    assert report.chosen.spec.name in ("cache_a", "cache_b")


# ------------------------------------------------------ never-engaged member

def test_never_engaged_member_rejects_combo_via_prefixed_signal():
    composer = TechniqueComposer(
        _jumpy_step, X0, 20, QualityBudget(max_rel_deviation=0.5))
    cache = _cache_composable("cache_jumpy", 0.05)
    counter = _counter_composable("counter")

    # direct probe: the prefixed signal makes the orchestrator's own
    # missing-signal check catch the member that never engaged
    verdict = composer.evaluate_combo([cache, counter])
    assert not verdict.accepted
    assert "never engaged" in verdict.reason
    assert "cache_jumpy.steps_reused" in verdict.reason

    # full search: the failed single is gated out, no combo is formed,
    # and the report surfaces the rejection reason
    report = composer.select([cache, counter])
    singles = {v.spec.name: v for v in report.singles}
    assert not singles["cache_jumpy"].accepted
    assert "never engaged" in singles["cache_jumpy"].reason
    assert report.combos == []
    assert report.chosen is not None
    assert report.chosen.spec.name == "counter"


# ------------------------------------------------------------- fail closed

def test_fail_closed_guards():
    composer = TechniqueComposer(
        _smooth_step, X0, 20, QualityBudget(max_rel_deviation=0.05))
    cache = _cache_composable("cache_g", 0.05)
    counter = _counter_composable("counter_g")

    for call in (lambda: composer.evaluate_singles([]),
                 lambda: composer.evaluate_combos([]),
                 lambda: composer.select([])):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("empty composables must be rejected")

    for call in (lambda: composer.evaluate_combos([cache, counter],
                                                  max_combo_size=1),
                 lambda: composer.select([cache, counter],
                                         max_combo_size=1)):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("max_combo_size < 2 must be rejected")

    try:
        composer.evaluate_combo([cache])
    except ValueError:
        pass
    else:
        raise AssertionError("a single-member combo must be rejected")

    try:
        composer.select([cache, _cache_composable("cache_g", 0.05)])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate names must be rejected")

    try:
        composer.select([cache, counter], margin=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative margin must be rejected")


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
