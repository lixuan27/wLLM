"""SLO compiler: service-level objectives as a first-class input.

An SLO is three things, and they act at three different moments:

* **hard constraints** — disqualify candidates outright (before and
  after measurement); a declared constraint that a candidate does not
  even report a measurement for is a violation, never a pass;
* **lifecycle** — startup cost is amortized over a replica's expected
  request count, so "faster steady-state" can lose to "faster to boot"
  for short-lived or serverless deployments;
* **soft preferences** — weights over latency / throughput / cost /
  startup / vram that rank the survivors; weights must sum to 1 so a
  spec cannot quietly over- or under-count an objective.

The compiler never outputs a single "global best" silently: it labels
the Pareto profiles (lowest latency / highest throughput / lowest
startup / lowest cost / strict exact) and then applies the preference
weights to choose a default, with the per-candidate score breakdown
kept as evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

HARD_KEYS = ("p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
             "peak_vram_gb", "startup_time_s", "quality_drop_max",
             "api_compatibility")
PREF_KEYS = ("latency", "throughput", "cost", "startup_time", "vram")

# hard-constraint key -> (CandidateMetrics attribute, comparison)
_HARD_METRIC = {
    "p50_latency_ms": ("p50_ms", "max"),
    "p95_latency_ms": ("p95_ms", "max"),
    "p99_latency_ms": ("p99_ms", "max"),
    "peak_vram_gb": ("peak_vram_gb", "max"),
    "startup_time_s": ("startup_s", "max"),
    "quality_drop_max": ("quality_drop", "max"),
}


@dataclass
class Lifecycle:
    expected_replica_lifetime_minutes: float = 240.0
    expected_requests_per_replica: int = 1000


@dataclass
class SLOSpec:
    hard_constraints: dict = field(default_factory=dict)
    preferences: dict = field(default_factory=dict)
    lifecycle: Lifecycle = field(default_factory=Lifecycle)

    def validate(self) -> list[str]:
        errs: list[str] = []
        for k, v in self.hard_constraints.items():
            if k not in HARD_KEYS:
                errs.append(f"unknown hard constraint {k!r}; "
                            f"choose from {HARD_KEYS}")
            elif k == "api_compatibility":
                if v not in ("strict", "relaxed"):
                    errs.append("api_compatibility must be strict|relaxed")
            elif not isinstance(v, (int, float)) or v <= 0:
                errs.append(f"hard constraint {k} must be positive, got {v!r}")
        for k, v in self.preferences.items():
            if k not in PREF_KEYS:
                errs.append(f"unknown preference {k!r}; "
                            f"choose from {PREF_KEYS}")
            elif not isinstance(v, (int, float)) or v < 0:
                errs.append(f"preference weight {k} must be >= 0")
        if self.preferences:
            total = sum(v for v in self.preferences.values()
                        if isinstance(v, (int, float)))
            if abs(total - 1.0) > 0.01:
                errs.append(f"preference weights must sum to 1.0 "
                            f"(got {total:.3f}); a lopsided sum silently "
                            f"re-weights every objective")
        if self.lifecycle.expected_requests_per_replica < 1:
            errs.append("lifecycle.expected_requests_per_replica must be >=1")
        if self.lifecycle.expected_replica_lifetime_minutes <= 0:
            errs.append("lifecycle.expected_replica_lifetime_minutes "
                        "must be > 0")
        return errs

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SLOSpec":
        known = {"hard_constraints", "preferences", "lifecycle"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown SLO fields: {sorted(unknown)}")
        return cls(hard_constraints=dict(d.get("hard_constraints") or {}),
                   preferences=dict(d.get("preferences") or {}),
                   lifecycle=Lifecycle(**(d.get("lifecycle") or {})))

    @classmethod
    def load(cls, path: str | Path) -> "SLOSpec":
        doc = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(doc, dict):
            raise ValueError(f"{path}: SLO root is not a mapping")
        return cls.from_dict(doc)


def amortized_seconds(startup_s: float, per_request_s: float,
                      n_requests: int, idle_s: float = 0.0) -> float:
    """Lifecycle cost per request: (startup + N*request + idle) / N."""
    if n_requests < 1:
        raise ValueError("n_requests must be >= 1")
    if min(startup_s, per_request_s, idle_s) < 0:
        raise ValueError("cost components must be >= 0")
    return (startup_s + n_requests * per_request_s + idle_s) / n_requests


@dataclass
class CandidateMetrics:
    name: str
    p50_ms: float
    p95_ms: float | None = None
    p99_ms: float | None = None
    throughput_rps: float | None = None
    peak_vram_gb: float | None = None
    startup_s: float | None = None
    cost_per_request: float | None = None
    quality_drop: float | None = None
    exact: bool = True

    def amortized_ms(self, lifecycle: Lifecycle) -> float | None:
        """Per-request latency with startup amortized over the lifecycle."""
        if self.startup_s is None:
            return None
        return amortized_seconds(
            self.startup_s, self.p50_ms / 1e3,
            lifecycle.expected_requests_per_replica) * 1e3


def admit(slo: SLOSpec, m: CandidateMetrics) -> list[str]:
    """Hard-constraint violations; empty == admitted.

    A declared constraint with no corresponding measurement on the
    candidate is a violation: unmeasured is not compliant.
    """
    violations: list[str] = []
    for key, limit in slo.hard_constraints.items():
        if key == "api_compatibility":
            continue           # enforced by the interface contract, not here
        attr, _ = _HARD_METRIC[key]
        val = getattr(m, attr, None)
        if val is None:
            violations.append(
                f"{m.name}: constraint {key} <= {limit} declared but the "
                f"candidate reports no {attr} measurement")
        elif val > limit:
            violations.append(
                f"{m.name}: {key} = {val} exceeds limit {limit}")
    return violations


def pareto_profiles(cands: list[CandidateMetrics]) -> dict[str, str]:
    """Label the classic profiles; only labels whose metric exists."""
    out: dict[str, str] = {}

    def best(label, key, reverse=False, pool=None):
        pool = [c for c in (pool if pool is not None else cands)
                if getattr(c, key) is not None]
        if pool:
            pick = (max if reverse else min)(pool, key=lambda c:
                                             getattr(c, key))
            out[label] = pick.name

    best("lowest_latency", "p50_ms")
    best("highest_throughput", "throughput_rps", reverse=True)
    best("lowest_startup", "startup_s")
    best("lowest_cost", "cost_per_request")
    best("strict_exact", "p50_ms", pool=[c for c in cands if c.exact])
    return out


# preference key -> (metric getter, lower_is_better)
def _pref_value(slo: SLOSpec, c: CandidateMetrics, key: str):
    if key == "latency":
        amort = c.amortized_ms(slo.lifecycle)
        return amort if amort is not None else c.p50_ms
    if key == "throughput":
        return c.throughput_rps
    if key == "cost":
        return c.cost_per_request
    if key == "startup_time":
        return c.startup_s
    if key == "vram":
        return c.peak_vram_gb
    raise KeyError(key)


@dataclass
class Selection:
    chosen: str | None
    scores: dict[str, float]
    rejected: dict[str, list[str]]
    profiles: dict[str, str]
    notes: list[str] = field(default_factory=list)


def choose(slo: SLOSpec, cands: list[CandidateMetrics]) -> Selection:
    """Admit by hard constraints, then rank by weighted preferences.

    Scores are min-max normalized per preference across the admitted
    set (lower is better for everything except throughput). A candidate
    missing a metric that carries preference weight gets the worst
    normalized value for it, and a note records the substitution —
    absence of measurement is never treated as being cheap.
    """
    errs = slo.validate()
    if errs:
        raise ValueError(f"invalid SLO: {errs}")
    if not cands:
        raise ValueError("no candidates to select from")
    rejected = {c.name: v for c in cands if (v := admit(slo, c))}
    admitted = [c for c in cands if c.name not in rejected]
    profiles = pareto_profiles(admitted)
    sel = Selection(chosen=None, scores={}, rejected=rejected,
                    profiles=profiles)
    if not admitted:
        sel.notes.append("no candidate satisfies the hard constraints")
        return sel
    prefs = slo.preferences or {"latency": 1.0}
    values: dict[str, dict[str, float | None]] = {
        key: {c.name: _pref_value(slo, c, key) for c in admitted}
        for key in prefs}
    for c in admitted:
        score = 0.0
        for key, weight in prefs.items():
            col = [v for v in values[key].values() if v is not None]
            v = values[key][c.name]
            if not col:
                continue       # nobody measured it; weight is inert
            lo, hi = min(col), max(col)
            if v is None:
                norm = 1.0
                sel.notes.append(
                    f"{c.name}: no {key} measurement; scored worst-case")
            elif hi == lo:
                norm = 0.0
            else:
                norm = (v - lo) / (hi - lo)
            if key == "throughput" and v is not None and hi != lo:
                norm = 1.0 - norm
            score += weight * norm
        sel.scores[c.name] = round(score, 6)
    sel.chosen = min(sel.scores, key=sel.scores.get)
    return sel
