"""Residual step cache for iterative refinement loops.

Iterative loops (denoise steps, refinement passes) often move slowly in
some step ranges: consecutive steps transform their input by nearly the
same residual. This technique reuses the previous step's residual when
the loop input changed less (relatively) than ``threshold``, skipping
the expensive step function entirely for that iteration.

Authenticity signal: ``steps_reused`` — zero reuse means the technique
never engaged and the orchestrator must reject the candidate rather
than report a fake win.

Numerics are plain Python lists of floats (torch-free) so the exact
same code paths are testable on any node; real deployments substitute
tensor operations behind the same interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

Vector = Sequence[float]


def _sub(a: Vector, b: Vector) -> list[float]:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    return [x - y for x, y in zip(a, b)]


def _add(a: Vector, b: Vector) -> list[float]:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    return [x + y for x, y in zip(a, b)]


def _norm(a: Vector) -> float:
    return math.sqrt(sum(x * x for x in a))


def should_reuse(delta_norm: float, base_norm: float, threshold: float,
                 consecutive: int, max_consecutive_reuses: int) -> bool:
    """The one reuse rule, shared by every backend of this technique.

    Both the torch-free reference implementation below and the tensor
    implementation used on real deployments call this predicate, so a
    simulated run and a hardware run can never drift apart on *when*
    the cache engages — only on the numerics they engage over.

    Non-finite norms (a diverged loop) never reuse: the cache must not
    extrapolate a residual it cannot trust.
    """
    if threshold <= 0:
        return False
    if consecutive >= max_consecutive_reuses:
        return False
    if not math.isfinite(delta_norm) or not math.isfinite(base_norm):
        return False
    return delta_norm / (base_norm + 1e-12) < threshold


@dataclass
class StepResidualCache:
    """Wraps ``step_fn(x, k) -> x'`` with residual-reuse skipping."""

    step_fn: Callable[[list[float], int], list[float]]
    threshold: float = 0.0          # 0 => never reuse (exact passthrough)
    # hard online guardrail: after this many consecutive reuses the step
    # function is recomputed no matter what, so a converging-tail input
    # can never make the cache extrapolate forever on a frozen residual
    max_consecutive_reuses: int = 4
    steps_total: int = 0
    steps_reused: int = 0
    _consecutive: int = field(default=0, repr=False)
    _last_input: list[float] | None = field(default=None, repr=False)
    _last_residual: list[float] | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.threshold < 0:
            raise ValueError("threshold must be >= 0")
        if self.max_consecutive_reuses < 1:
            raise ValueError("max_consecutive_reuses must be >= 1")

    def reset(self) -> None:
        """Full per-request reset: cache state AND evidence counters.

        Create one cache per evaluation, or reset between requests —
        stale counters would let a never-engaged run pass the
        authenticity check on a previous run's evidence.
        """
        self._last_input = None
        self._last_residual = None
        self._consecutive = 0
        self.steps_total = 0
        self.steps_reused = 0

    def __call__(self, x: list[float], k: int) -> list[float]:
        self.steps_total += 1
        if (self.threshold > 0 and self._last_input is not None
                and self._last_residual is not None):
            if should_reuse(_norm(_sub(x, self._last_input)),
                            _norm(self._last_input), self.threshold,
                            self._consecutive, self.max_consecutive_reuses):
                self.steps_reused += 1
                self._consecutive += 1
                out = _add(x, self._last_residual)
                self._last_input = list(x)
                return out
        out = self.step_fn(list(x), k)
        self._last_residual = _sub(out, x)
        self._last_input = list(x)
        self._consecutive = 0
        return out

    def authenticity(self) -> dict[str, float]:
        return {"steps_total": float(self.steps_total),
                "steps_reused": float(self.steps_reused)}


def run_loop(step: Callable[[list[float], int], list[float]],
             x0: Vector, iterations: int) -> list[float]:
    """Drive an iterative loop; shared by reference and cached runs."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    x = list(x0)
    for k in range(iterations):
        x = step(x, k)
    return x
