"""Numerical verification (wBench level B).

Compares candidate outputs against reference outputs under the program's
quality contract.  Pure-python allclose supporting nested dict/list/tuple
of floats, ints, and objects exposing ``tolist()`` (numpy/torch tensors),
so the verifier itself has zero heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..graph.quality import QualityContract


@dataclass
class VerifyResult:
    passed: bool
    checked: int
    mismatches: list[str] = field(default_factory=list)

    def report(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[numerical] {status} ({self.checked} leaves compared)"]
        lines += [f"  - {m}" for m in self.mismatches[:20]]
        return "\n".join(lines)


def _leaves(obj: Any, path: str = "$"):
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    if isinstance(obj, dict):
        for key in sorted(obj):
            yield from _leaves(obj[key], f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            yield from _leaves(item, f"{path}[{i}]")
    else:
        yield path, obj


def compare(reference: Any, candidate: Any,
            contract: QualityContract) -> VerifyResult:
    atol = contract.latent_atol
    ref_leaves = dict(_leaves(reference))
    cand_leaves = dict(_leaves(candidate))
    mismatches: list[str] = []

    for path in sorted(set(ref_leaves) | set(cand_leaves)):
        if path not in ref_leaves:
            mismatches.append(f"{path}: extra in candidate")
            continue
        if path not in cand_leaves:
            mismatches.append(f"{path}: missing in candidate")
            continue
        r, c = ref_leaves[path], cand_leaves[path]
        if isinstance(r, float) or isinstance(c, float):
            if abs(float(r) - float(c)) > atol:
                mismatches.append(f"{path}: |{r} - {c}| > atol={atol}")
        elif r != c:
            mismatches.append(f"{path}: {r!r} != {c!r}")

    return VerifyResult(passed=not mismatches,
                        checked=len(ref_leaves),
                        mismatches=mismatches)
