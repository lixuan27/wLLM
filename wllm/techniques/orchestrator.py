"""Technique orchestration: reference vs candidates, fail-closed.

The orchestrator owns the comparison protocol so no technique can grade
itself:

1. run the exact reference once on the shared inputs;
2. run each candidate in isolation on the *same* inputs;
3. measure element-wise relative deviation against the reference;
4. reject any candidate that crashed, exceeded the quality budget, or
   failed to produce its declared authenticity signals;
5. rank survivors by measured wall time and emit receipt-ready dicts.

A candidate can never win by being wrong, by not engaging, or by only
claiming to be fast.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .base import QualityBudget, TechniqueResult, TechniqueSpec


@dataclass
class CandidateVerdict:
    spec: TechniqueSpec
    accepted: bool
    reason: str
    max_rel_deviation: float | None = None
    speedup: float | None = None
    authenticity: dict[str, float] = field(default_factory=dict)
    wall_ms: float | None = None

    def receipt_fields(self) -> dict:
        return {
            "plan_id": f"tech-{self.spec.name}",
            "passes": [self.spec.name],
            "quality": {
                "verdict": ("exact" if self.max_rel_deviation == 0.0
                            else "bounded" if self.accepted else "failed"),
                "max_rel_deviation": self.max_rel_deviation,
            },
            "authenticity": {k: v > 0 for k, v in self.authenticity.items()},
            "perf": ({"p50_ms": self.wall_ms, "p95_ms": self.wall_ms}
                     if self.wall_ms else {}),
        }


def max_rel_deviation(ref: Sequence[float], cand: Sequence[float],
                      eps: float = 1e-8) -> float:
    if len(ref) != len(cand):
        raise ValueError(f"output length mismatch: reference {len(ref)} "
                         f"vs candidate {len(cand)}")
    worst = 0.0
    for r, c in zip(ref, cand):
        worst = max(worst, abs(c - r) / (abs(r) + eps))
    return worst


@dataclass
class TechniqueOrchestrator:
    """``runner(technique_ctx) -> (outputs, authenticity)``.

    ``reference`` computes exact outputs; each candidate supplies a
    spec and a runner closure over the same inputs. ``repeats`` timing
    runs use the median to damp scheduler noise — both closures are
    therefore executed ``repeats`` times and MUST be pure: construct any
    stateful object (e.g. a step cache) inside the closure, never share
    one across calls, or warm state pollutes timing and authenticity.
    """

    reference: Callable[[], Sequence[float]]
    budget: QualityBudget = field(default_factory=QualityBudget.exact)
    repeats: int = 3

    def evaluate(self, candidates: list[tuple[TechniqueSpec, Callable[[], tuple]]]
                 ) -> list[CandidateVerdict]:
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1")
        ref_out, ref_ms = self._timed(self.reference)
        if not ref_out:
            raise ValueError(
                "reference produced empty outputs; an empty oracle would "
                "accept every empty candidate as exact — refusing to grade")
        verdicts: list[CandidateVerdict] = []
        for spec, runner in candidates:
            spec_errs = spec.validate()
            if spec_errs:
                verdicts.append(CandidateVerdict(
                    spec, False, f"invalid spec: {spec_errs}"))
                continue
            try:
                (cand_out, authenticity), cand_ms = self._timed_pair(runner)
            except Exception as exc:  # noqa: BLE001 — crash == rejection
                verdicts.append(CandidateVerdict(
                    spec, False, f"crashed: {type(exc).__name__}: {exc}"))
                continue
            result = TechniqueResult(spec=spec, outputs=cand_out,
                                     authenticity=dict(authenticity),
                                     wall_ms=cand_ms)
            missing = result.missing_signals()
            if missing:
                verdicts.append(CandidateVerdict(
                    spec, False,
                    f"authenticity signals absent or zero: {missing} "
                    f"(optimization never engaged)",
                    authenticity=result.authenticity, wall_ms=cand_ms))
                continue
            try:
                dev = max_rel_deviation(ref_out, cand_out, self.budget.eps)
            except ValueError as exc:
                verdicts.append(CandidateVerdict(
                    spec, False, f"output shape violation: {exc}",
                    authenticity=result.authenticity, wall_ms=cand_ms))
                continue
            if dev > self.budget.max_rel_deviation:
                verdicts.append(CandidateVerdict(
                    spec, False,
                    f"quality budget exceeded: deviation {dev:.3e} > "
                    f"budget {self.budget.max_rel_deviation:.3e}",
                    max_rel_deviation=dev,
                    authenticity=result.authenticity, wall_ms=cand_ms))
                continue
            verdicts.append(CandidateVerdict(
                spec, True, "accepted", max_rel_deviation=dev,
                speedup=(ref_ms / cand_ms) if cand_ms > 0 else None,
                authenticity=result.authenticity, wall_ms=cand_ms))
        accepted = [v for v in verdicts if v.accepted]
        rejected = [v for v in verdicts if not v.accepted]
        accepted.sort(key=lambda v: v.wall_ms if v.wall_ms is not None
                      else float("inf"))
        return accepted + rejected

    # ------------------------------------------------------------- timing
    def _timed(self, fn: Callable[[], Sequence[float]]):
        out, times = None, []
        for _ in range(self.repeats):
            t0 = time.perf_counter()
            out = fn()
            times.append((time.perf_counter() - t0) * 1e3)
        return out, sorted(times)[len(times) // 2]

    def _timed_pair(self, fn: Callable[[], tuple]):
        out, times = None, []
        for _ in range(self.repeats):
            t0 = time.perf_counter()
            out = fn()
            times.append((time.perf_counter() - t0) * 1e3)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise TypeError("candidate runner must return "
                            "(outputs, authenticity_dict)")
        return out, sorted(times)[len(times) // 2]
