"""Quality contracts: exact by default, approximate only when budgeted.

The planner's transformation registry is split into exact and approximate
tables; approximate transformations may be considered only when the
program's quality contract is BOUNDED_DEGRADATION and the specific budget
keys they affect are present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class QualityMode(str, Enum):
    EXACT = "exact"
    BOUNDED_DEGRADATION = "bounded_degradation"


@dataclass
class QualityContract:
    mode: QualityMode = QualityMode.EXACT
    seed: int | None = 1234           # fixed seed for exact comparisons
    # exact-mode tolerances (documented, explicit):
    latent_atol: float = 1e-5
    logits_atol: float = 1e-4
    # bounded-degradation budgets, e.g. {"vbench_drop_max": 0.005,
    #   "temporal_lpips_delta_max": 0.01, "success_rate_drop_max": 0.0,
    #   "deadline_miss_rate_max": 0.001, "safety_violations_max": 0}
    budgets: dict[str, float] = field(default_factory=dict)

    def allows_approximate(self) -> bool:
        return self.mode == QualityMode.BOUNDED_DEGRADATION

    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.mode == QualityMode.BOUNDED_DEGRADATION and not self.budgets:
            errs.append("bounded_degradation requires at least one budget entry")
        if self.mode == QualityMode.EXACT and self.budgets:
            errs.append("exact mode must not carry degradation budgets")
        for key, val in self.budgets.items():
            if val < 0:
                errs.append(f"budget '{key}' must be >= 0")
        return errs
