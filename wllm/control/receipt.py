"""Receipts: no measured evidence, no optimization claim.

Every candidate that ran carries one receipt. Promotion (apply) is gated
on `promote_problems()` being empty: performance must be a distribution
from a real run, authenticity checks must all pass (the optimization was
really active), no forbidden log pattern fired, and the quality verdict
must match the spec's policy. The deployment fingerprint pins the exact
world the numbers came from; any key-field change invalidates reuse.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

FINGERPRINT_FIELDS = (
    "plan_id", "backend", "backend_version", "source_revision",
    "model_revision", "hardware", "driver", "torch_version", "precision",
    "passes",
)

REQUIRED_PERF_KEYS = ("p50_ms", "p95_ms")


@dataclass
class Receipt:
    plan_id: str
    backend: str
    backend_version: str = ""
    source_revision: str = ""
    model_revision: str = ""
    hardware: str = ""
    driver: str = ""
    torch_version: str = ""
    precision: str = ""
    passes: list[str] = field(default_factory=list)
    # measured evidence -----------------------------------------------------
    perf: dict = field(default_factory=dict)       # p50_ms/p95_ms/p99_ms/...
    baseline_perf: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)    # verdict: exact|bounded|failed
    authenticity: dict = field(default_factory=dict)   # check -> bool
    fallback_hits: list[str] = field(default_factory=list)
    # bookkeeping -----------------------------------------------------------
    known_limitations: list[str] = field(default_factory=list)
    rollback_target: str = "reference"
    created_unix: float = field(default_factory=time.time)

    # ---------------------------------------------------------- fingerprint
    def fingerprint(self) -> str:
        basis = {k: getattr(self, k) for k in FINGERPRINT_FIELDS}
        blob = json.dumps(basis, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    # ------------------------------------------------------------- promote
    def promote_problems(self, quality_policy: str = "exact") -> list[str]:
        """Empty list == this receipt may be applied. Fail closed otherwise."""
        errs: list[str] = []
        for k in REQUIRED_PERF_KEYS:
            v = self.perf.get(k)
            if not isinstance(v, (int, float)) or v <= 0:
                errs.append(f"perf.{k} missing or non-positive "
                            f"(claims without measurement are void)")
        failing = sorted(k for k, ok in self.authenticity.items() if not ok)
        if failing:
            errs.append(f"authenticity checks failed: {failing} "
                        f"(optimization not proven active)")
        if not self.authenticity:
            errs.append("no authenticity checks recorded "
                        "(cannot prove the optimization was active)")
        if self.fallback_hits:
            errs.append(f"forbidden log patterns fired: {self.fallback_hits} "
                        f"(silent fallback invalidates the measurement)")
        verdict = self.quality.get("verdict")
        if verdict is None:
            errs.append("quality verdict missing")
        elif quality_policy == "exact" and verdict != "exact":
            errs.append(f"quality verdict {verdict!r} not acceptable under "
                        f"exact policy")
        elif verdict == "failed":
            errs.append("quality verdict is 'failed'")
        if not self.backend:
            errs.append("backend missing")
        return errs

    def speedup(self) -> float | None:
        base = self.baseline_perf.get("p50_ms")
        cand = self.perf.get("p50_ms")
        if isinstance(base, (int, float)) and isinstance(cand, (int, float)) \
                and cand > 0:
            return base / cand
        return None

    # ------------------------------------------------------------ round-trip
    def to_dict(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        return d

    def save(self, receipts_dir: str | Path) -> Path:
        out = Path(receipts_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.plan_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Receipt":
        d = json.loads(Path(path).read_text())
        stored_fp = d.pop("fingerprint", None)
        rec = cls(**d)
        if stored_fp and stored_fp != rec.fingerprint():
            raise ValueError(
                f"{path}: fingerprint mismatch (stored {stored_fp}, "
                f"recomputed {rec.fingerprint()}) — receipt tampered or "
                f"schema drifted; refusing to trust it")
        return rec
