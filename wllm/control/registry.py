"""Backend capability registry: experience as machine-checkable data.

Every backend ships a declarative capability file instead of prose
advice. The planner consults `legal_passes` (requires/conflicts under
the active quality policy, with rejection reasons) and the verifier
consults `scan_log` (fail-closed invariants: a forbidden log pattern —
e.g. a silent fallback to a slower path — invalidates the whole
measurement, no matter how good the numbers look).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

QUALITY_LEVELS = ("exact", "bounded", "experimental")


@dataclass
class PassCap:
    name: str
    quality: str = "exact"                    # exact | bounded | experimental
    requires: dict = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class BackendCap:
    backend: str
    version: str = ""
    models_exact: list[str] = field(default_factory=list)
    models_compatible: list[str] = field(default_factory=list)
    modalities_in: list[str] = field(default_factory=list)
    modalities_out: list[str] = field(default_factory=list)
    passes: dict[str, PassCap] = field(default_factory=dict)
    forbidden_log_patterns: list[str] = field(default_factory=list)

    def supports_model(self, model_id: str) -> str | None:
        """'exact' | 'compatible' | None."""
        if model_id in self.models_exact:
            return "exact"
        for pat in self.models_compatible:
            if re.fullmatch(_glob_to_re(pat), model_id):
                return "compatible"
        return None


def _glob_to_re(pat: str) -> str:
    return "".join(".*" if ch == "*" else re.escape(ch) for ch in pat)


def parse_capability(doc: dict, origin: str = "<dict>") -> BackendCap:
    if not isinstance(doc, dict) or "backend" not in doc:
        raise ValueError(f"{origin}: capability file needs a 'backend' field")
    passes: dict[str, PassCap] = {}
    for name, spec in (doc.get("passes") or {}).items():
        spec = spec or {}
        quality = spec.get("quality", "exact")
        if quality not in QUALITY_LEVELS:
            raise ValueError(
                f"{origin}: pass {name!r} has unknown quality {quality!r}")
        passes[name] = PassCap(
            name=name, quality=quality,
            requires=spec.get("requires") or {},
            conflicts=list(spec.get("conflicts") or []),
            notes=str(spec.get("notes") or ""),
        )
    models = doc.get("models") or {}
    inv = doc.get("invariants") or {}
    return BackendCap(
        backend=str(doc["backend"]),
        version=str(doc.get("version") or ""),
        models_exact=list(models.get("exact") or []),
        models_compatible=list(models.get("compatible") or []),
        modalities_in=list((doc.get("modalities") or {}).get("input") or []),
        modalities_out=list((doc.get("modalities") or {}).get("output") or []),
        passes=passes,
        forbidden_log_patterns=list(inv.get("forbidden_log_patterns") or []),
    )


def load_registry(directory: str | Path) -> dict[str, BackendCap]:
    caps: dict[str, BackendCap] = {}
    for f in sorted(Path(directory).glob("*.yaml")):
        cap = parse_capability(yaml.safe_load(f.read_text()) or {}, str(f))
        caps[cap.backend] = cap
    return caps


def default_registry() -> dict[str, BackendCap]:
    return load_registry(Path(__file__).parent / "registry_data")


# ---------------------------------------------------------------- selection

def rank_backends(caps: dict[str, BackendCap], model_id: str,
                  required_out: list[str] | None = None,
                  ) -> list[tuple[BackendCap, str]]:
    """Backends able to serve the model, exact matches first."""
    ranked: list[tuple[BackendCap, str]] = []
    for cap in caps.values():
        tier = cap.supports_model(model_id)
        if tier is None:
            continue
        if required_out and not set(required_out) <= set(cap.modalities_out):
            continue
        ranked.append((cap, tier))
    ranked.sort(key=lambda ct: (0 if ct[1] == "exact" else 1, ct[0].backend))
    return ranked


# --------------------------------------------------------------- pass logic

@dataclass
class PassDecision:
    name: str
    kept: bool
    reason: str


def legal_passes(cap: BackendCap, context: dict, quality_policy: str = "exact",
                 requested: list[str] | None = None) -> list[PassDecision]:
    """Filter the backend's passes under context + quality policy.

    ``context`` supplies facts like {"num_gpus": 2, "model_uses_cfg": True}.
    A `requires` entry of the form {fact: expected} keeps the pass only if
    the context fact equals the expectation; numeric facts support
    {"min_<fact>": n}. Unknown facts fail closed (rejected, with reason).
    """
    names = requested if requested is not None else list(cap.passes)
    decisions: list[PassDecision] = []
    kept_names: set[str] = set()
    for name in names:
        p = cap.passes.get(name)
        if p is None:
            decisions.append(PassDecision(name, False,
                                          "unknown pass for this backend"))
            continue
        if quality_policy == "exact" and p.quality != "exact":
            decisions.append(PassDecision(
                name, False,
                f"quality={p.quality} not allowed under exact policy"))
            continue
        problem = _requires_problem(p.requires, context)
        if problem:
            decisions.append(PassDecision(name, False, problem))
            continue
        decisions.append(PassDecision(name, True, "ok"))
        kept_names.add(name)
    # conflicts resolved after individual checks: first-kept wins
    order = [d.name for d in decisions if d.kept]
    for d in decisions:
        if not d.kept:
            continue
        p = cap.passes[d.name]
        earlier = order[:order.index(d.name)]
        clash = [c for c in p.conflicts if c in earlier]
        if clash:
            d.kept = False
            d.reason = f"conflicts with already-selected {clash[0]}"
            kept_names.discard(d.name)
    return decisions


def _requires_problem(requires: dict, context: dict) -> str:
    for key, expected in (requires or {}).items():
        if key.startswith("min_"):
            fact = key[4:]
            val = context.get(fact)
            if val is None:
                return f"requires {fact} but context does not state it"
            if not isinstance(val, (int, float)) or val < expected:
                return f"requires {fact} >= {expected}, context has {val}"
        else:
            val = context.get(key)
            if val is None:
                return f"requires {key} but context does not state it"
            if val != expected:
                return f"requires {key} == {expected!r}, context has {val!r}"
    return ""


# ------------------------------------------------------- fail-closed scans

@dataclass
class LogScan:
    patterns_scanned: int
    hits: list[str]

    @property
    def invalidated(self) -> bool:
        return bool(self.hits)


def scan_log(cap: BackendCap, log_text: str) -> LogScan:
    """A single forbidden-pattern hit invalidates the measurement."""
    hits = [pat for pat in cap.forbidden_log_patterns
            if re.search(pat, log_text, flags=re.IGNORECASE)]
    return LogScan(patterns_scanned=len(cap.forbidden_log_patterns), hits=hits)
