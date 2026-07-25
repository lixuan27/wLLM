"""Model profile schema: per-model compatibility contracts.

A profile is not documentation.  Every claim carries an evidence
pointer (a SLURM job log, a saved receipt, or a report section) and
every profile carries a binding -- backend version, hardware, and a
``last_validated`` date -- so profiles expire instead of rotting
silently.  Validation is fail-closed in the spirit of
``wllm.control.spec``: unknown enum values, measured-level claims
without evidence refs, and malformed dates reject the profile with
reasons instead of being silently defaulted.
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass, field

EVIDENCE_KINDS = ("job_log", "receipt", "report")
QUALITY_CLASSES = ("exact", "bounded", "experimental")
# Honest support ladder, weakest to strongest claim.
SUPPORT_TIERS = ("discovered", "cataloged", "launchable",
                 "parity-verified", "optimized", "serving-verified")
# Tiers from "launchable" up assert something actually ran, so a
# measured-level profile must show evidence for them.
_TIERS_NEEDING_EVIDENCE = ("launchable", "parity-verified",
                           "optimized", "serving-verified")
EVIDENCE_LEVELS = ("measured", "reported", "declared")

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _glob_to_re(pat: str) -> str:
    """Same '*' translation the control registry uses for compatible
    model globs; reimplemented locally rather than importing a
    private helper from another package."""
    return "".join(".*" if ch == "*" else re.escape(ch) for ch in pat)


def _parses_as_date(s: str) -> bool:
    try:
        datetime.date.fromisoformat(s)
        return True
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------ leaf types

@dataclass
class EvidenceRef:
    """Pointer to on-disk proof: a job log, a receipt, or a report."""
    kind: str
    ref: str
    note: str = ""


@dataclass
class OptimizationEntry:
    """One optimization claim; a measured claim without evidence refs
    is invalid (see ``ModelProfile.validate``)."""
    name: str
    quality_class: str = "exact"
    evidence: list[EvidenceRef] = field(default_factory=list)
    notes: str = ""


@dataclass
class RuntimeSupport:
    """Where the model runs, and how strong the claim honestly is."""
    runtime: str
    tier: str = "discovered"
    evidence: list[EvidenceRef] = field(default_factory=list)


@dataclass
class ProfileBinding:
    """What the profile's claims were validated against, and when."""
    backend_version: str = ""
    hardware: str = ""
    cuda: str = ""
    last_validated: str = ""          # YYYY-MM-DD, mandatory
    evidence_level: str = "declared"  # measured | reported | declared


# ----------------------------------------------------- dict conversions

def _evidence_from_dict(d: dict) -> EvidenceRef:
    d = dict(d or {})
    ev = EvidenceRef(kind=str(d.pop("kind", "")),
                     ref=str(d.pop("ref", "")),
                     note=str(d.pop("note", "")))
    if d:
        raise ValueError(f"unknown evidence fields: {sorted(d)}")
    return ev


def _opt_from_dict(d: dict) -> OptimizationEntry:
    d = dict(d or {})
    entry = OptimizationEntry(
        name=str(d.pop("name", "")),
        quality_class=str(d.pop("quality_class", "exact")),
        evidence=[_evidence_from_dict(e)
                  for e in d.pop("evidence", None) or []],
        notes=str(d.pop("notes", "")),
    )
    if d:
        raise ValueError(f"unknown optimization fields: {sorted(d)}")
    return entry


def _runtime_from_dict(d: dict) -> RuntimeSupport:
    d = dict(d or {})
    rs = RuntimeSupport(
        runtime=str(d.pop("runtime", "")),
        tier=str(d.pop("tier", "discovered")),
        evidence=[_evidence_from_dict(e)
                  for e in d.pop("evidence", None) or []],
    )
    if d:
        raise ValueError(f"unknown runtime_support fields: {sorted(d)}")
    return rs


def _evidence_problems(evidence: list[EvidenceRef],
                       where: str) -> list[str]:
    errs: list[str] = []
    for ev in evidence:
        if ev.kind not in EVIDENCE_KINDS:
            errs.append(f"{where}: unknown evidence kind {ev.kind!r}; "
                        f"choose one of {EVIDENCE_KINDS}")
        if not ev.ref:
            errs.append(f"{where}: evidence ref is empty")
    return errs


# ---------------------------------------------------------- the profile

@dataclass
class ModelProfile:
    """Machine-executable compatibility contract for a model family.

    Constraints enforced by ``validate`` (fail-closed):
    - detection_ids must be non-empty (a profile that can never
      match a model id is dead weight);
    - tiers, quality classes, evidence kinds, and evidence levels
      must come from the closed vocabularies above;
    - a "measured" binding demands evidence refs on every
      optimization entry and on every runtime tier that asserts
      something actually ran;
    - ``last_validated`` must be a real YYYY-MM-DD date, because
      profiles expire (see ``is_stale``).
    """
    model_family: str = ""
    detection_ids: list[str] = field(default_factory=list)
    components: list[str] = field(default_factory=list)
    runtime_support: list[RuntimeSupport] = field(default_factory=list)
    optimizations_lossless: list[OptimizationEntry] = \
        field(default_factory=list)
    optimizations_bounded: list[OptimizationEntry] = \
        field(default_factory=list)
    incompatibilities: list[str] = field(default_factory=list)
    authenticity_signals: list[str] = field(default_factory=list)
    validation: list[str] = field(default_factory=list)
    binding: ProfileBinding = field(default_factory=ProfileBinding)

    # -------------------------------------------------------- validation
    def validate(self) -> list[str]:
        """Return a list of problems; empty means valid."""
        errs: list[str] = []
        if not self.model_family:
            errs.append("model_family is empty")
        if not self.detection_ids:
            errs.append("detection_ids is empty; a profile that can "
                        "never match a model id is dead weight")
        for rs in self.runtime_support:
            where = f"runtime {rs.runtime!r}"
            if not rs.runtime:
                errs.append("runtime_support entry has empty runtime")
            if rs.tier not in SUPPORT_TIERS:
                errs.append(f"{where}: unknown tier {rs.tier!r}; "
                            f"choose one of {SUPPORT_TIERS}")
            errs += _evidence_problems(rs.evidence, where)
        for entry in self.optimizations_lossless:
            errs += self._opt_problems(entry, "optimizations_lossless")
        for entry in self.optimizations_bounded:
            errs += self._opt_problems(entry, "optimizations_bounded")
        for inc in self.incompatibilities:
            if " + " not in inc:
                errs.append(f"incompatibility {inc!r} must look like "
                            "'passA + passB'")
        b = self.binding
        if b.evidence_level not in EVIDENCE_LEVELS:
            errs.append(f"unknown evidence level {b.evidence_level!r}; "
                        f"choose one of {EVIDENCE_LEVELS}")
        if not (_DATE_RE.fullmatch(b.last_validated or "")
                and _parses_as_date(b.last_validated)):
            errs.append(f"binding.last_validated {b.last_validated!r} "
                        "is not a real YYYY-MM-DD date")
        if b.evidence_level == "measured":
            errs += self._measured_problems()
        return errs

    def _opt_problems(self, entry: OptimizationEntry,
                      bucket: str) -> list[str]:
        errs: list[str] = []
        where = f"{bucket}: {entry.name!r}"
        if not entry.name:
            errs.append(f"{bucket}: optimization with empty name")
        if entry.quality_class not in QUALITY_CLASSES:
            errs.append(f"{where}: unknown quality_class "
                        f"{entry.quality_class!r}; choose one of "
                        f"{QUALITY_CLASSES}")
        elif (bucket == "optimizations_lossless"
                and entry.quality_class != "exact"):
            errs.append(f"{where}: quality_class "
                        f"{entry.quality_class!r} must not sit in the "
                        "lossless list; move it to "
                        "optimizations_bounded")
        elif (bucket == "optimizations_bounded"
                and entry.quality_class == "exact"):
            errs.append(f"{where}: exact entries belong in "
                        "optimizations_lossless")
        errs += _evidence_problems(entry.evidence, where)
        return errs

    def _measured_problems(self) -> list[str]:
        errs: list[str] = []
        for entry in (self.optimizations_lossless
                      + self.optimizations_bounded):
            if not entry.evidence:
                errs.append(
                    f"measured profile: optimization {entry.name!r} "
                    "has no evidence refs; a measured claim without "
                    "evidence is invalid")
        for rs in self.runtime_support:
            if rs.tier in _TIERS_NEEDING_EVIDENCE and not rs.evidence:
                errs.append(
                    f"measured profile: runtime {rs.runtime!r} at "
                    f"tier {rs.tier!r} has no evidence refs")
        return errs

    # --------------------------------------------------------- staleness
    def is_stale(self, today: str, max_age_days: int = 90) -> bool:
        """True when the binding is older than ``max_age_days``.

        ``today`` is an explicit YYYY-MM-DD parameter; this method
        never reads the wall clock, so staleness checks stay
        reproducible.  Unparseable dates count as stale
        (fail-closed).
        """
        try:
            now = datetime.date.fromisoformat(today)
            seen = datetime.date.fromisoformat(
                self.binding.last_validated)
        except (TypeError, ValueError):
            return True
        return (now - seen).days > max_age_days

    # ---------------------------------------------------------- matching
    def matches(self, model_id: str) -> bool:
        """Exact detection id, or glob ('*' wildcard) full match."""
        for pat in self.detection_ids:
            if pat == model_id:
                return True
            if "*" in pat and re.fullmatch(_glob_to_re(pat), model_id):
                return True
        return False

    # -------------------------------------------------------- round-trip
    @classmethod
    def from_dict(cls, d: dict,
                  origin: str = "<dict>") -> "ModelProfile":
        known = {"model_family", "detection_ids", "components",
                 "runtime_support", "optimizations_lossless",
                 "optimizations_bounded", "incompatibilities",
                 "authenticity_signals", "validation", "binding"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"{origin}: unknown profile fields: {sorted(unknown)}")
        return cls(
            model_family=str(d.get("model_family") or ""),
            detection_ids=[str(x)
                           for x in d.get("detection_ids") or []],
            components=[str(x) for x in d.get("components") or []],
            runtime_support=[_runtime_from_dict(x)
                             for x in d.get("runtime_support") or []],
            optimizations_lossless=[
                _opt_from_dict(x)
                for x in d.get("optimizations_lossless") or []],
            optimizations_bounded=[
                _opt_from_dict(x)
                for x in d.get("optimizations_bounded") or []],
            incompatibilities=[
                str(x) for x in d.get("incompatibilities") or []],
            authenticity_signals=[
                str(x) for x in d.get("authenticity_signals") or []],
            validation=[str(x) for x in d.get("validation") or []],
            binding=ProfileBinding(**(d.get("binding") or {})),
        )
