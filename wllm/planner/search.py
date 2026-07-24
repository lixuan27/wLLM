"""Measurement-driven plan search: successive halving under a budget.

The cost model orders candidate plans; measurement decides winners.
Successive halving spends the measurement budget geometrically: every
surviving plan gets a short probe, losers are culled, survivors get a
longer run, until the Pareto set gets a full-length validation.  This
replaces any fixed "measure everything for N seconds" rule and lets the
user cap total spend (`budget_s`).

The searcher is execution-agnostic: the caller supplies `measure(plan,
duration_s) -> Measurement`, so the same loop drives toy CPU runs, srun
probes, or full sbatch benchmarks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .plan import DeploymentPlan


@dataclass
class Measurement:
    plan_id: str
    duration_s: float
    ok: bool
    latency_ms: float | None = None        # first-output latency (median)
    sustained_rate: float | None = None    # units/s at steady state
    p95_gap_ms: float | None = None        # smoothness side condition
    error: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class SearchRecord:
    plan: DeploymentPlan
    rounds: list[Measurement] = field(default_factory=list)
    culled_at_round: int | None = None
    cull_reason: str = ""

    @property
    def last(self) -> Measurement | None:
        return self.rounds[-1] if self.rounds else None


@dataclass
class SearchResult:
    records: list[SearchRecord]
    objective: str
    spent_s: float

    def survivors(self) -> list[SearchRecord]:
        return [r for r in self.records
                if r.culled_at_round is None and r.rounds and r.last.ok]

    def best(self) -> SearchRecord | None:
        surv = self.survivors()
        if not surv:
            return None
        return min(surv, key=lambda r: _score(r.last, self.objective))

    def report(self) -> str:
        lines = [f"search objective={self.objective} spent={self.spent_s:.0f}s"]
        for rec in self.records:
            if rec.culled_at_round is not None:
                lines.append(f"  culled r{rec.culled_at_round} {rec.plan.id}: "
                             f"{rec.cull_reason}")
            elif rec.last and rec.last.ok:
                lines.append(
                    f"  alive {rec.plan.id}: latency={rec.last.latency_ms} "
                    f"rate={rec.last.sustained_rate} p95gap={rec.last.p95_gap_ms}")
            else:
                lines.append(f"  failed {rec.plan.id}: "
                             f"{rec.last.error if rec.last else 'never measured'}")
        return "\n".join(lines)


def _score(m: Measurement, objective: str) -> float:
    if objective == "sustained-rate":
        return -(m.sustained_rate or 0.0)
    return m.latency_ms if m.latency_ms is not None else float("inf")


def successive_halving(
    plans: list[DeploymentPlan],
    measure: Callable[[DeploymentPlan, float], Measurement],
    *,
    objective: str = "first-output-latency",
    probe_s: float = 30.0,
    growth: float = 3.0,
    keep_fraction: float = 0.5,
    min_final: int = 2,
    budget_s: float | None = None,
) -> SearchResult:
    """Run rounds of measure-and-cull until <=min_final plans remain or the
    budget is exhausted.  Failed measurements are culled with the error as
    the recorded reason (a failure is a result, not an exception)."""
    records = {p.id: SearchRecord(plan=p) for p in plans}
    alive = list(plans)
    spent = 0.0
    duration = probe_s
    round_idx = 0

    while len(alive) > max(min_final, 1):
        round_idx += 1
        results: list[tuple[DeploymentPlan, Measurement]] = []
        for plan in alive:
            if budget_s is not None and spent >= budget_s:
                break
            t0 = time.monotonic()
            try:
                m = measure(plan, duration)
            except Exception as exc:  # noqa: BLE001 — searcher must survive
                m = Measurement(plan_id=plan.id, duration_s=duration,
                                ok=False, error=repr(exc))
            spent += max(time.monotonic() - t0, 0.0)
            records[plan.id].rounds.append(m)
            results.append((plan, m))

        ok_results = [(p, m) for p, m in results if m.ok]
        for plan, m in results:
            if not m.ok:
                records[plan.id].culled_at_round = round_idx
                records[plan.id].cull_reason = f"measurement failed: {m.error}"
        if not ok_results:
            alive = []
            break

        ok_results.sort(key=lambda pm: _score(pm[1], objective))
        n_keep = max(min_final, int(len(ok_results) * keep_fraction))
        keep = {p.id for p, _ in ok_results[:n_keep]}
        for plan, m in ok_results[n_keep:]:
            records[plan.id].culled_at_round = round_idx
            records[plan.id].cull_reason = (
                f"outperformed at round {round_idx} "
                f"(score {_score(m, objective):.1f})")
        alive = [p for p, _ in ok_results if p.id in keep]

        if budget_s is not None and spent >= budget_s:
            break
        if len(ok_results) <= max(min_final, 1):
            break
        duration *= growth

    return SearchResult(records=list(records.values()), objective=objective,
                        spent_s=spent)
