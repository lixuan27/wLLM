"""Technique contracts: specs, results, and quality budgets."""

from __future__ import annotations

from dataclasses import dataclass, field

FAMILIES = ("cache", "quantization", "sparsity", "fusion", "parallelism",
            "scheduling")
QUALITY_CLASSES = ("exact", "bounded", "experimental")


@dataclass
class TechniqueSpec:
    name: str
    family: str
    quality_class: str = "bounded"
    params: dict = field(default_factory=dict)
    # signal name -> predicate description; all must be non-trivially met
    authenticity_signals: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        errs = []
        if self.family not in FAMILIES:
            errs.append(f"unknown technique family {self.family!r}")
        if self.quality_class not in QUALITY_CLASSES:
            errs.append(f"unknown quality class {self.quality_class!r}")
        if not self.authenticity_signals:
            errs.append(f"technique {self.name!r} declares no authenticity "
                        f"signals; unverifiable optimizations are not "
                        f"admissible")
        return errs


@dataclass
class QualityBudget:
    """Explicit deviation budget for bounded techniques.

    ``max_rel_deviation`` bounds ``|cand - ref| / (|ref| + eps)`` at the
    element level (max over elements). ``exact`` budgets are the zero
    budget.
    """
    max_rel_deviation: float = 0.0
    eps: float = 1e-8

    @classmethod
    def exact(cls) -> "QualityBudget":
        return cls(max_rel_deviation=0.0)


@dataclass
class TechniqueResult:
    spec: TechniqueSpec
    outputs: object
    authenticity: dict[str, float] = field(default_factory=dict)
    wall_ms: float = 0.0
    notes: str = ""

    def missing_signals(self) -> list[str]:
        """Declared signals that were not reported or reported as zero."""
        out = []
        for name in self.spec.authenticity_signals:
            val = self.authenticity.get(name)
            if val is None or val <= 0:
                out.append(name)
        return out
