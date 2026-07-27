"""Tensor backends for the reuse-cache technique family.

The reference implementation in :mod:`wllm.techniques.step_cache` is
torch-free so its decision logic can be exercised anywhere.  Real
deployments iterate over tensors, and re-deriving the reuse rule for
them would let the simulated and the deployed technique drift apart —
the classic way a technique "wins" in the harness and misbehaves in
production.  So these backends import the *same* predicate
(:func:`~wllm.techniques.step_cache.should_reuse`) and differ only in
the algebra they perform.

Two members, and picking the wrong one is a correctness bug, not a
tuning choice:

``TensorStepResidualCache``
    Reuses the *state update* of an iterative loop: ``x' = x + residual``.
    Skipping an iteration therefore skips the loop's own bookkeeping.
    **Only legal when the loop is memoryless** — a one-step solver whose
    next state depends on the current state alone.

``TensorOutputReuseCache``
    Reuses the *output of an expensive function* re-evaluated on a
    slowly changing input, while the surrounding loop still runs every
    iteration.  This is the member required whenever the loop carries
    state the cache must not desynchronize — for example a multistep
    ODE solver, which keeps a history of previous model outputs and a
    step index, so skipping its update would silently corrupt the
    trajectory rather than approximate it.

torch is imported lazily inside the methods, so importing this module
costs nothing on a node without it; both caches work on CPU tensors,
which is how they are unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .step_cache import should_reuse


@dataclass
class _TensorReuseCacheBase:
    """Shared state, evidence and engagement rule for both backends."""

    step_fn: Callable[[Any, int], Any]
    threshold: float = 0.0
    max_consecutive_reuses: int = 4
    steps_total: int = 0
    steps_reused: int = 0
    # per-step relative input deltas actually observed; this is the
    # evidence a threshold choice must be justified against, rather
    # than a constant somebody guessed
    deltas: list[float] = field(default_factory=list)
    _consecutive: int = field(default=0, repr=False)
    _last_input: Any = field(default=None, repr=False)
    _cached: Any = field(default=None, repr=False)

    def __post_init__(self):
        if self.threshold < 0:
            raise ValueError("threshold must be >= 0")
        if self.max_consecutive_reuses < 1:
            raise ValueError("max_consecutive_reuses must be >= 1")

    def reset(self) -> None:
        """Per-request reset of cache state AND evidence counters."""
        self._last_input = None
        self._cached = None
        self._consecutive = 0
        self.steps_total = 0
        self.steps_reused = 0
        self.deltas = []

    def _engages(self, x) -> bool:
        """Measure the input's relative move and apply the shared rule.

        Norms are taken in float32 whatever the working dtype: a bf16
        norm over a large latent loses enough mantissa to make the
        decision itself noisy.
        """
        import torch

        with torch.no_grad():
            delta = torch.linalg.vector_norm(
                (x - self._last_input).float()).item()
            base = torch.linalg.vector_norm(self._last_input.float()).item()
        self.deltas.append(delta / (base + 1e-12))
        return should_reuse(delta, base, self.threshold, self._consecutive,
                            self.max_consecutive_reuses)

    def authenticity(self) -> dict[str, float]:
        """Evidence the orchestrator checks before believing a win.

        ``steps_reused == 0`` means the technique never engaged, so any
        measured speedup came from somewhere else and the candidate must
        be rejected rather than credited to this pass.
        """
        return {"steps_total": float(self.steps_total),
                "steps_reused": float(self.steps_reused)}


@dataclass
class TensorStepResidualCache(_TensorReuseCacheBase):
    """``step_fn(x, k) -> x'`` over tensors, reusing the state update.

    Legal only for memoryless loops — see the module docstring.
    """

    def __call__(self, x, k: int):
        self.steps_total += 1
        if self._last_input is not None and self._cached is not None:
            if self._engages(x):
                self.steps_reused += 1
                self._consecutive += 1
                out = x + self._cached
                self._last_input = x.clone()
                return out
        out = self.step_fn(x, k)
        self._cached = (out - x).clone()
        self._last_input = x.clone()
        self._consecutive = 0
        return out


@dataclass
class TensorOutputReuseCache(_TensorReuseCacheBase):
    """``step_fn(x, k) -> f(x)`` over tensors, reusing the output itself.

    The surrounding loop keeps running every iteration; only the
    expensive evaluation is skipped, so stateful solvers stay in sync.
    """

    def __call__(self, x, k: int):
        self.steps_total += 1
        if self._last_input is not None and self._cached is not None:
            if self._engages(x):
                self.steps_reused += 1
                self._consecutive += 1
                self._last_input = x.clone()
                return self._cached
        out = self.step_fn(x, k)
        self._cached = out.clone()
        self._last_input = x.clone()
        self._consecutive = 0
        return out
