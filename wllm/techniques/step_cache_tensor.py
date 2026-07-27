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
    """Shared state, evidence and engagement rule for both backends.

    ``key`` selects *which* signal the reuse rule is applied to, and the
    choice is load-bearing rather than cosmetic:

    ``"input"``
        Reuse while the loop's input is barely moving.  Cheap, but it is
        a proxy: an input can move slowly while the expensive function's
        value moves fast, and then the cache engages in exactly the
        phase where reuse is least safe.
    ``"output"``
        Reuse while the *function's own value* is barely moving, judged
        between its last two genuine evaluations.  Self-correcting — the
        moment the output starts changing, reuse stops — at the cost of
        needing two real evaluations before it can engage at all.
    """

    step_fn: Callable[[Any, int], Any]
    threshold: float = 0.0
    max_consecutive_reuses: int = 4
    key: str = "input"
    steps_total: int = 0
    steps_reused: int = 0
    # per-step relative input deltas actually observed; this is the
    # evidence a threshold choice must be justified against, rather
    # than a constant somebody guessed
    deltas: list[float] = field(default_factory=list)
    _consecutive: int = field(default=0, repr=False)
    _last_input: Any = field(default=None, repr=False)
    _cached: Any = field(default=None, repr=False)
    _prev_eval: Any = field(default=None, repr=False)
    _out_move: tuple | None = field(default=None, repr=False)

    KEYS = ("input", "output")

    def __post_init__(self):
        if self.threshold < 0:
            raise ValueError("threshold must be >= 0")
        if self.max_consecutive_reuses < 1:
            raise ValueError("max_consecutive_reuses must be >= 1")
        if self.key not in self.KEYS:
            raise ValueError(f"key must be one of {self.KEYS}, got {self.key!r}")

    def reset(self) -> None:
        """Per-request reset of cache state AND evidence counters."""
        self._last_input = None
        self._cached = None
        self._prev_eval = None
        self._out_move = None
        self._consecutive = 0
        self.steps_total = 0
        self.steps_reused = 0
        self.deltas = []

    @staticmethod
    def _relative_move(a, b) -> tuple:
        """``(||a - b||, ||b||)`` in float32, whatever the working dtype.

        A bf16 norm over a large tensor loses enough mantissa to make
        the reuse decision itself noisy, so the norms are always widened.
        """
        import torch

        with torch.no_grad():
            return (torch.linalg.vector_norm((a - b).float()).item(),
                    torch.linalg.vector_norm(b.float()).item())

    def _engages(self, x) -> bool:
        """Apply the shared rule to whichever signal ``key`` selects."""
        if self.key == "input":
            delta, base = self._relative_move(x, self._last_input)
        else:
            if self._out_move is None:
                return False      # not two genuine evaluations yet
            delta, base = self._out_move
        self.deltas.append(delta / (base + 1e-12))
        return should_reuse(delta, base, self.threshold, self._consecutive,
                            self.max_consecutive_reuses)

    def _record_evaluation(self, out) -> None:
        """Remember how far the function's value moved since last time."""
        if self.key == "output":
            if self._prev_eval is not None:
                self._out_move = self._relative_move(out, self._prev_eval)
            self._prev_eval = out.clone()

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
        # one guard, not two: _cached and _last_input are written in the
        # same tail below, so testing both would be a condition no test
        # could ever exercise independently
        if self._cached is not None:
            if self._engages(x):
                self.steps_reused += 1
                self._consecutive += 1
                out = x + self._cached
                self._last_input = x.clone()
                return out
        out = self.step_fn(x, k)
        self._record_evaluation(out)
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
        # one guard, not two: _cached and _last_input are written in the
        # same tail below, so testing both would be a condition no test
        # could ever exercise independently
        if self._cached is not None:
            if self._engages(x):
                self.steps_reused += 1
                self._consecutive += 1
                self._last_input = x.clone()
                return self._cached
        out = self.step_fn(x, k)
        self._record_evaluation(out)
        self._cached = out.clone()
        self._last_input = x.clone()
        self._consecutive = 0
        return out
