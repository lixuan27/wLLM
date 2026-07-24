"""Typed OptimizeSpec: the optimizer reads this, never natural language.

The agent bridge translates a sentence like "optimize this project on
4 GPUs, first-frame latency first, no quality loss" into this schema and
nothing else. Validation is fail-closed: an invalid spec is rejected
with reasons instead of being silently defaulted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

PRIMARY_OBJECTIVES = (
    "p50_e2e_latency", "p95_e2e_latency", "p95_first_output",
    "first_output_latency", "sustained_rate", "throughput",
    "gpu_seconds", "deadline_miss_rate",
)
QUALITY_POLICIES = ("exact", "bounded")
BUDGET_PRESETS = ("quick", "balanced", "thorough")
MODALITIES = ("text", "image", "video", "audio", "action", "latent")


@dataclass
class HardwareSpec:
    accelerator: str = "auto"
    count: int | str = "auto"


@dataclass
class ObjectiveSpec:
    primary: str = "p95_first_output"
    secondary: list[str] = field(default_factory=list)


@dataclass
class QualitySpec:
    policy: str = "exact"
    # bounded policy requires an explicit budget: {"metric": ..., "max": ...}
    budget: dict | None = None


@dataclass
class ContractSpec:
    preserve_existing_api: bool = True
    required_modalities: list[str] = field(default_factory=list)


@dataclass
class OptimizeSpec:
    project: str = "."
    hardware: HardwareSpec = field(default_factory=HardwareSpec)
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    quality: QualitySpec = field(default_factory=QualitySpec)
    contract: ContractSpec = field(default_factory=ContractSpec)
    budget: str | dict = "balanced"

    # ------------------------------------------------------------ validation
    def validate(self) -> list[str]:
        """Return a list of problems; empty means valid."""
        errs: list[str] = []
        if self.objective.primary not in PRIMARY_OBJECTIVES:
            errs.append(f"unknown primary objective {self.objective.primary!r}; "
                        f"choose one of {PRIMARY_OBJECTIVES}")
        for s in self.objective.secondary:
            if s not in PRIMARY_OBJECTIVES:
                errs.append(f"unknown secondary objective {s!r}")
        if self.quality.policy not in QUALITY_POLICIES:
            errs.append(f"unknown quality policy {self.quality.policy!r}; "
                        f"choose one of {QUALITY_POLICIES}")
        if self.quality.policy == "bounded":
            b = self.quality.budget
            if not (isinstance(b, dict) and "metric" in b and "max" in b):
                errs.append("bounded quality policy requires an explicit "
                            "budget {metric, max}; refusing to guess one")
        if self.quality.policy == "exact" and self.quality.budget:
            errs.append("exact quality policy must not carry a budget "
                        "(ambiguous intent)")
        if isinstance(self.budget, str):
            if self.budget not in BUDGET_PRESETS:
                errs.append(f"unknown budget preset {self.budget!r}")
        elif isinstance(self.budget, dict):
            gh = self.budget.get("gpu_hours")
            if not isinstance(gh, (int, float)) or gh <= 0:
                errs.append("dict budget requires positive gpu_hours")
        else:
            errs.append(f"budget must be preset or dict, got "
                        f"{type(self.budget).__name__}")
        for m in self.contract.required_modalities:
            if m not in MODALITIES:
                errs.append(f"unknown modality {m!r}")
        if isinstance(self.hardware.count, int) and self.hardware.count <= 0:
            errs.append("hardware.count must be positive")
        elif isinstance(self.hardware.count, str) and \
                self.hardware.count != "auto":
            errs.append("hardware.count must be an int or 'auto'")
        return errs

    # ------------------------------------------------------------ round-trip
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        return p

    @classmethod
    def from_dict(cls, d: dict) -> "OptimizeSpec":
        known = {"project", "hardware", "objective", "quality", "contract",
                 "budget"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown OptimizeSpec fields: {sorted(unknown)}")
        return cls(
            project=d.get("project", "."),
            hardware=HardwareSpec(**(d.get("hardware") or {})),
            objective=ObjectiveSpec(**(d.get("objective") or {})),
            quality=QualitySpec(**(d.get("quality") or {})),
            contract=ContractSpec(**(d.get("contract") or {})),
            budget=d.get("budget", "balanced"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "OptimizeSpec":
        doc = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(doc, dict):
            raise ValueError(f"{path}: spec root is not a mapping")
        return cls.from_dict(doc)
