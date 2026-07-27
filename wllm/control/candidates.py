"""Integrated candidate planning: three knowledge sources, one funnel.

Candidates are generated from the backend capability registry, then
pruned by everything the system has *learned*:

* the model profile (verified contract) supplies the legal
  optimization sets and known-incompatible combinations — a stale
  profile downgrades trust instead of being silently believed;
* the optimization trace store supplies known-dead configurations,
  skipped WITH their recorded reason and trace id;
* the SLO supplies hard constraints, declared up front so every
  surviving candidate knows what admission will demand after
  measurement — planning never pretends to check what only a
  measurement can.

Every keep and every rejection carries provenance. The output is a
plan of *candidates to measure*, never a claim of speedup.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .registry import BackendCap, legal_passes, rank_backends
from .slo import SLOSpec


@dataclass
class CandidateSpec:
    backend: str
    passes: list[str]
    support: str                     # exact | compatible (registry tier)
    provenance: list[str] = field(default_factory=list)

    @property
    def candidate_key(self) -> dict:
        """The trace-store candidate identity for this spec."""
        return {"backend": self.backend, "passes": sorted(self.passes)}


@dataclass
class PlanningReport:
    model_id: str
    mode: str                        # "plan" | "diagnose-only"
    candidates: list[CandidateSpec] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)   # key -> reason
    pending_gates: list[str] = field(default_factory=list)   # SLO constraints
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "mode": self.mode,
            "candidates": [{"backend": c.backend, "passes": c.passes,
                            "support": c.support,
                            "provenance": c.provenance}
                           for c in self.candidates],
            "rejected": self.rejected,
            "pending_gates": self.pending_gates,
            "notes": self.notes,
        }


def _profile_pass_sets(profile, quality_policy: str,
                       notes: list[str]) -> tuple[set[str], set[str]]:
    """(allowed passes per profile, bounded-only passes)."""
    lossless = {o.name for o in profile.optimizations_lossless}
    bounded = {o.name for o in profile.optimizations_bounded}
    if quality_policy == "exact":
        if bounded:
            notes.append(
                f"profile lists bounded passes {sorted(bounded)}; "
                f"excluded under exact policy")
        return lossless, set()
    return lossless | bounded, bounded


def _incompatibility_prune(passes: list[str], incompatibilities: list[str],
                           provenance: list[str]) -> list[str]:
    """Drop the later member of any profiled-incompatible pair."""
    kept = list(passes)
    for rule in incompatibilities:
        parts = [p.strip() for p in rule.split(" + ")]
        if len(parts) != 2:
            continue
        a, b = parts
        if a in kept and b in kept:
            kept.remove(b)
            provenance.append(
                f"dropped pass {b!r}: profile marks '{rule}' incompatible")
    return kept


def plan_candidates(model_id: str, *, hardware: str, context: dict,
                    quality_policy: str = "exact",
                    registry: dict[str, BackendCap],
                    profiles: dict | None = None,
                    trace_store=None,
                    slo: SLOSpec | None = None,
                    today: str | None = None) -> PlanningReport:
    """Build the measured-candidate plan from all knowledge sources.

    ``profiles`` maps family -> ModelProfile (see wllm.profiles);
    ``trace_store`` is a wllm.control.tracestore.TraceStore;
    ``today`` (YYYY-MM-DD) enables profile staleness checks — omitted
    means staleness is not evaluated (and a note records that).
    """
    report = PlanningReport(model_id=model_id, mode="plan")

    profile = None
    if profiles:
        from ..profiles import match as match_profile
        profile = match_profile(profiles, model_id)
    profile_fresh = False
    if profile is None:
        report.notes.append(
            "no verified profile matches this model; only registry "
            "knowledge applies" if profiles else
            "no profile pack supplied")
    elif today is None:
        report.notes.append(
            f"profile {profile.model_family!r} matched but freshness is "
            f"unverified (no reference date); its evidence distinction "
            f"is NOT applied — incompatibility rules still are")
    elif profile.is_stale(today):
        report.notes.append(
            f"profile {profile.model_family!r} is STALE (last validated "
            f"{profile.binding.last_validated}); its evidence "
            f"distinction is NOT applied (enforced) — incompatibility "
            f"rules, being warnings, still are")
    else:
        profile_fresh = True

    if slo is not None:
        errs = slo.validate()
        if errs:
            raise ValueError(f"invalid SLO: {errs}")
        report.pending_gates = [
            f"{k} <= {v}" for k, v in sorted(slo.hard_constraints.items())
            if k != "api_compatibility"]
        if slo.hard_constraints.get("api_compatibility") == "strict":
            report.pending_gates.append("api_compatibility == strict")

    if trace_store is not None:
        hw_seen = {t.hardware for t in trace_store.all()}
        if hw_seen and hardware not in hw_seen:
            report.notes.append(
                f"hardware {hardware!r} has no recorded traces (store "
                f"covers {sorted(hw_seen)}); trace pruning is inert "
                f"for this plan")

    allowed_by_profile: set[str] | None = None
    if profile is not None and profile_fresh:
        allowed_by_profile, _ = _profile_pass_sets(
            profile, quality_policy, report.notes)

    ranked = rank_backends(registry, model_id)
    for cap, support in ranked:
        decisions = legal_passes(cap, context, quality_policy)
        kept = [d.name for d in decisions if d.kept]
        for d in decisions:
            if not d.kept and d.reason != "ok":
                report.rejected[f"{cap.backend}:{d.name}"] = d.reason
        provenance = [f"registry: backend {cap.backend} supports model "
                      f"({support}); {len(kept)} legal passes under "
                      f"{quality_policy} policy"]
        if allowed_by_profile is not None:
            unknown_to_profile = [p for p in kept
                                  if p not in allowed_by_profile]
            if unknown_to_profile:
                provenance.append(
                    f"passes {unknown_to_profile} lack profile evidence "
                    f"for this model; kept as registry-only claims")
        if profile is not None:
            # incompatibility rules are warnings of measured danger;
            # they apply regardless of profile freshness
            kept = _incompatibility_prune(
                kept, profile.incompatibilities, provenance)
        if not kept:
            report.rejected[cap.backend] = (
                "no legal pass survives pruning for this backend")
            continue
        spec = CandidateSpec(backend=cap.backend, passes=kept,
                             support=support, provenance=provenance)
        if trace_store is not None:
            # workload comes from the context when the caller stated one;
            # without it the store deliberately answers conservatively,
            # because an acceptance measured on one workload is not
            # evidence about another
            workload = context.get("workload")
            bad = trace_store.known_bad(model_id, hardware,
                                        spec.candidate_key, workload)
            if bad is not None:
                report.rejected[cap.backend] = (
                    f"known-bad from trace {bad.trace_id} "
                    f"({bad.recorded}): {bad.reason}")
                continue
            # per-pass history: skip individual passes recorded dead
            surviving = []
            for p in spec.passes:
                pbad = trace_store.known_bad(
                    model_id, hardware, {"pass": p, "gpus":
                                         context.get("num_gpus", 1)},
                    workload)
                if pbad is not None:
                    provenance.append(
                        f"dropped pass {p!r}: known-bad from trace "
                        f"{pbad.trace_id} ({pbad.recorded}): {pbad.reason}")
                else:
                    surviving.append(p)
            spec.passes = surviving
            if not spec.passes:
                report.rejected[cap.backend] = (
                    "every pass for this backend is recorded known-bad")
                continue
        report.candidates.append(spec)

    if not any(c.passes for c in report.candidates):
        report.mode = "diagnose-only"
        report.notes.append(
            "no optimizing candidate survives; the reference path "
            "remains available, nothing will be changed")
    return report
