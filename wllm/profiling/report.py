"""Baseline profiler: measured evidence or it didn't happen.

Times any zero-argument thunk over warmup+measured repeats, reports
median/p95/min/max wall time, and persists a JSON evidence file under
benchmarks/results/.  GPU memory stats are attached when torch with CUDA
is importable; absence is recorded explicitly rather than silently.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class BaselineReport:
    name: str
    repeats: int
    warmup: int
    wall_ms: list[float] = field(default_factory=list)
    median_ms: float = 0.0
    p95_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    gpu_mem_peak_bytes: int | None = None
    gpu_available: bool = False
    meta: dict = field(default_factory=dict)

    def finalize(self) -> "BaselineReport":
        if self.wall_ms:
            xs = sorted(self.wall_ms)
            self.median_ms = statistics.median(xs)
            idx = min(len(xs) - 1, max(0, round(0.95 * (len(xs) - 1))))
            self.p95_ms = xs[idx]
            self.min_ms = xs[0]
            self.max_ms = xs[-1]
        return self

    def save(self, out_dir: str | Path, tag: str = "") -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = out / f"{self.name}{('_' + tag) if tag else ''}_{stamp}.json"
        path.write_text(json.dumps(asdict(self), indent=1))
        return path


def _gpu_probe():
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            return torch
    except Exception:  # noqa: BLE001
        pass
    return None


def profile_thunk(name: str, thunk: Callable[[], object], *,
                  repeats: int = 5, warmup: int = 1,
                  meta: dict | None = None) -> BaselineReport:
    torch = _gpu_probe()
    report = BaselineReport(name=name, repeats=repeats, warmup=warmup,
                            gpu_available=torch is not None,
                            meta=meta or {})
    for _ in range(warmup):
        thunk()
    if torch is not None:
        torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        if torch is not None:
            torch.cuda.synchronize()
        t0 = time.monotonic()
        thunk()
        if torch is not None:
            torch.cuda.synchronize()
        report.wall_ms.append((time.monotonic() - t0) * 1000.0)
    if torch is not None:
        report.gpu_mem_peak_bytes = int(torch.cuda.max_memory_allocated())
    return report.finalize()
