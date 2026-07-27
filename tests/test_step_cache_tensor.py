"""Tensor reuse caches: same engagement rule as the reference backend.

The point of these tests is not that the caches "work" — it is that the
tensor backends and the torch-free reference backend can never disagree
about *when* the technique engages, and that the guardrails (consecutive
reuse cap, non-finite input, threshold 0) hold on real tensors.
"""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wllm.techniques.step_cache import StepResidualCache, should_reuse
from wllm.techniques.step_cache_tensor import (TensorOutputReuseCache,
                                               TensorStepResidualCache)


def _drift_sequence(n: int, shrink: float):
    """Inputs whose step-to-step relative move shrinks geometrically."""
    xs, x = [], torch.ones(8, dtype=torch.float32)
    for k in range(n):
        xs.append(x.clone())
        x = x + torch.full((8,), shrink ** (k + 1))
    return xs


# --------------------------------------------------------- shared predicate

def test_predicate_boundary_and_fail_closed():
    # clearly inside / clearly outside the threshold
    assert should_reuse(0.9, 10.0, 0.1, 0, 4) is True
    assert should_reuse(1.1, 10.0, 0.1, 0, 4) is False
    # the comparison is strict, but the 1e-12 guard added to the
    # denominator puts a nominally exact ratio a hair *inside* the
    # threshold; pinned so the guard cannot be dropped unnoticed
    assert should_reuse(1.0, 10.0, 0.1, 0, 4) is True
    # a degenerate zero-magnitude state must never reuse: the relative
    # move is astronomically large, so the cache falls back to a real
    # recomputation instead of dividing by nothing
    assert should_reuse(0.5, 0.0, 0.1, 0, 4) is False
    # the consecutive cap is inclusive: at the cap, recompute
    assert should_reuse(0.0, 10.0, 0.1, 3, 4) is True
    assert should_reuse(0.0, 10.0, 0.1, 4, 4) is False
    # threshold 0 disables the technique entirely
    assert should_reuse(0.0, 10.0, 0.0, 0, 4) is False
    # a diverged loop must never ride a stale residual
    assert should_reuse(float("nan"), 10.0, 0.5, 0, 4) is False
    assert should_reuse(1.0, float("inf"), 0.5, 0, 4) is False


# ------------------------------------------------- backend agreement on rule

def test_tensor_and_reference_backends_engage_identically():
    seq = _drift_sequence(12, 0.5)
    calls_ref, calls_ten = [], []

    ref = StepResidualCache(step_fn=lambda x, k: (calls_ref.append(k)
                                                  or [v * 1.5 for v in x]),
                            threshold=0.05)
    ten = TensorStepResidualCache(
        step_fn=lambda x, k: (calls_ten.append(k) or x * 1.5), threshold=0.05)
    # drive both over the same inputs: the caches only decide, the caller
    # supplies the trajectory, so the comparison is apples to apples
    for k, x in enumerate(seq):
        ref(list(x.tolist()), k)
        ten(x, k)

    assert ref.steps_reused == ten.steps_reused > 0, (
        ref.steps_reused, ten.steps_reused)
    assert ref.steps_total == ten.steps_total == len(seq)
    assert calls_ref == calls_ten          # identical recompute schedule


# ------------------------------------------------------- tensor cache rules

def test_output_reuse_returns_the_cached_output_verbatim():
    n_calls = []

    def f(x, k):
        n_calls.append(k)
        return x * 3.0 + k

    cache = TensorOutputReuseCache(step_fn=f, threshold=0.5)
    x0 = torch.ones(4)
    first = cache(x0, 0)
    # a tiny move must reuse: same object value, no new evaluation
    second = cache(x0 + 1e-6, 1)
    assert torch.equal(first, second)
    assert n_calls == [0]
    assert cache.steps_reused == 1 and cache.steps_total == 2
    # a large move must recompute
    third = cache(x0 * 10, 2)
    assert not torch.equal(third, second)
    assert n_calls == [0, 2]


def test_residual_reuse_extrapolates_the_last_update():
    cache = TensorStepResidualCache(step_fn=lambda x, k: x + 2.0,
                                    threshold=0.5)
    x0 = torch.full((4,), 10.0)
    out0 = cache(x0, 0)
    assert torch.equal(out0, torch.full((4,), 12.0))
    x1 = torch.full((4,), 10.001)             # relative move 1e-4
    out1 = cache(x1, 1)                       # reuse: x1 + residual(=2)
    assert torch.allclose(out1, x1 + 2.0)
    assert cache.steps_reused == 1


def test_tensor_cache_never_reuses_from_a_zero_state():
    """The zero-base degenerate case, on the tensor path.

    Starting from an all-zero state, any move is infinitely large in
    relative terms, so the cache must recompute rather than divide by
    (almost) nothing and extrapolate a residual it cannot justify.
    """
    n_calls = []
    cache = TensorStepResidualCache(
        step_fn=lambda x, k: (n_calls.append(k) or x + 2.0), threshold=0.5)
    cache(torch.zeros(4), 0)
    cache(torch.full((4,), 1e-3), 1)
    assert cache.steps_reused == 0 and n_calls == [0, 1]


def test_consecutive_cap_forces_recomputation():
    n_calls = []
    cache = TensorOutputReuseCache(
        step_fn=lambda x, k: (n_calls.append(k) or x * 2.0),
        threshold=1.0, max_consecutive_reuses=2)
    x = torch.ones(4)
    for k in range(6):
        cache(x + k * 1e-9, k)
    # steps 1,2 reuse; step 3 is forced; 4,5 reuse again
    assert n_calls == [0, 3], n_calls
    assert cache.steps_reused == 4


def test_threshold_zero_is_exact_passthrough():
    n_calls = []
    cache = TensorOutputReuseCache(
        step_fn=lambda x, k: (n_calls.append(k) or x.clone()), threshold=0.0)
    x = torch.ones(4)
    for k in range(5):
        cache(x, k)
    assert cache.steps_reused == 0
    assert n_calls == [0, 1, 2, 3, 4]


def test_non_finite_input_never_reuses():
    n_calls = []
    cache = TensorOutputReuseCache(
        step_fn=lambda x, k: (n_calls.append(k) or torch.ones(4)),
        threshold=1.0)
    cache(torch.ones(4), 0)
    cache(torch.full((4,), float("nan")), 1)
    assert cache.steps_reused == 0 and n_calls == [0, 1]


def test_reset_clears_state_and_evidence():
    cache = TensorOutputReuseCache(step_fn=lambda x, k: x * 2.0,
                                   threshold=1.0)
    x = torch.ones(4)
    cache(x, 0)
    cache(x, 1)
    assert cache.steps_reused == 1 and cache.deltas
    cache.reset()
    assert (cache.steps_total, cache.steps_reused, cache.deltas) == (0, 0, [])
    # a fresh request must recompute rather than trust the old cache
    n_calls = []
    cache.step_fn = lambda x, k: (n_calls.append(k) or x * 2.0)
    cache(x, 0)
    assert n_calls == [0] and cache.steps_reused == 0


def test_authenticity_and_deltas_are_reported():
    cache = TensorOutputReuseCache(step_fn=lambda x, k: x, threshold=1.0)
    x = torch.ones(4)
    cache(x, 0)
    cache(x * 1.5, 1)
    ev = cache.authenticity()
    assert ev["steps_total"] == 2.0
    # the first call has no predecessor, so exactly one delta is observed
    assert len(cache.deltas) == 1
    assert abs(cache.deltas[0] - 0.5) < 1e-6


def test_output_key_waits_for_two_real_evaluations():
    """The output key cannot judge stability it has not observed twice."""
    n_calls = []
    cache = TensorOutputReuseCache(
        step_fn=lambda x, k: (n_calls.append(k) or torch.ones(4)),
        threshold=1.0, key="output")
    x = torch.ones(4)
    cache(x, 0)                     # first evaluation: nothing to compare
    assert cache.steps_reused == 0
    cache(x, 1)                     # second: still no prior *move* known
    assert cache.steps_reused == 0 and n_calls == [0, 1]
    cache(x, 2)                     # now the function is known to be flat
    assert cache.steps_reused == 1 and n_calls == [0, 1]


def test_output_key_ignores_a_still_input_when_the_output_moves():
    """The whole point: a quiet input must not license reuse.

    The input never changes here, so the input key would reuse forever;
    the output key sees the function's value jumping and refuses.
    """
    outs = iter([torch.zeros(4), torch.full((4, ), 100.0),
                 torch.zeros(4), torch.full((4, ), 100.0)])
    n_calls = []
    cache = TensorOutputReuseCache(
        step_fn=lambda x, k: (n_calls.append(k) or next(outs)),
        threshold=0.1, key="output")
    x = torch.ones(4)
    for k in range(4):
        cache(x, k)
    assert cache.steps_reused == 0, cache.deltas
    assert n_calls == [0, 1, 2, 3]


def test_input_key_would_reuse_where_the_output_key_refuses():
    """Same trace, opposite verdicts — the keys are not interchangeable."""
    outs = iter([torch.zeros(4), torch.full((4,), 100.0),
                 torch.zeros(4), torch.full((4,), 100.0)])
    n_calls = []
    cache = TensorOutputReuseCache(
        step_fn=lambda x, k: (n_calls.append(k) or next(outs)),
        threshold=0.1, key="input")
    x = torch.ones(4)
    for k in range(4):
        cache(x, k)
    assert cache.steps_reused > 0 and len(n_calls) < 4


def test_defaults_are_the_conservative_ones():
    """A cache built with no arguments must be inert and bounded.

    Defaults are a safety contract, not cosmetics: a technique that
    engages before anyone asked for it would silently change results in
    every pipeline that constructs one without thinking.
    """
    n_calls = []
    for cls in (TensorOutputReuseCache, TensorStepResidualCache):
        n_calls.clear()
        cache = cls(step_fn=lambda x, k: (n_calls.append(k) or x * 2.0))
        x = torch.ones(4)
        for k in range(4):
            cache(x, k)
        assert cache.steps_reused == 0, f"{cls.__name__} reuses by default"
        assert n_calls == [0, 1, 2, 3]
        assert cache.key == "input"          # the cheap key is the default
        assert cache.max_consecutive_reuses == 4

    # and the default cap really is the cap: with reuse always allowed,
    # every 5th step is a forced recomputation
    n_calls.clear()
    capped = TensorOutputReuseCache(
        step_fn=lambda x, k: (n_calls.append(k) or x * 2.0), threshold=1.0)
    x = torch.ones(4)
    for k in range(11):
        capped(x, k)
    assert n_calls == [0, 5, 10], n_calls


def test_reference_backend_defaults_match_the_tensor_backend():
    """The two backends must not disagree on their default contract."""
    calls = []
    ref = StepResidualCache(step_fn=lambda x, k: (calls.append(k)
                                                  or [v * 2 for v in x]))
    for k in range(4):
        ref([1.0, 1.0], k)
    assert ref.steps_reused == 0 and calls == [0, 1, 2, 3]
    assert ref.max_consecutive_reuses == 4
    assert ref.threshold == TensorOutputReuseCache(
        step_fn=lambda x, k: x).threshold


def test_unknown_key_is_rejected():
    try:
        TensorOutputReuseCache(step_fn=lambda x, k: x, key="vibes")
    except ValueError as exc:
        assert "key must be one of" in str(exc)
    else:
        raise AssertionError("unknown reuse keys must be rejected")


def test_invalid_construction_is_rejected():
    for kwargs in ({"threshold": -0.1}, {"max_consecutive_reuses": 0}):
        try:
            TensorOutputReuseCache(step_fn=lambda x, k: x, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"must reject {kwargs}")


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
