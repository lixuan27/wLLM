"""Turn on-disk measured evidence into promotable receipts.

Benchmark jobs leave three kinds of evidence: phase-labelled benchmark
JSON (median + raw times per phase), single-line parity JSON records
(``{"pair": ..., "max_abs": ..., "bitexact": ...}``), and the raw job
log. This module parses that evidence into a :class:`Receipt` so the
promote gate operates on what actually ran — a receipt built here is
only as good as the evidence: missing phases, absent parity checks, or
a forbidden log pattern all surface as promote blockers, never as
defaults.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field

from .receipt import Receipt
from .registry import BackendCap, scan_log

_PHASE_RE = re.compile(r"^==\s*phase\s+(\d+):\s*(.+?)\s*$", re.MULTILINE)
_MEDIAN_RE = re.compile(r'"median_ms":\s*([0-9.]+)')
_PARITY_RE = re.compile(r'^\{"pair".*\}$', re.MULTILINE)


@dataclass
class PhaseLogEvidence:
    phases: dict[str, float] = field(default_factory=dict)  # name -> median_ms
    times_ms: dict[str, list[float]] = field(default_factory=dict)
    # raw text of each phase's log chunk: authenticity checks must grep
    # EXECUTION markers here (what actually ran), never trust the label
    phase_text: dict[str, str] = field(default_factory=dict)
    parity: list[dict] = field(default_factory=list)
    gate_markers: list[str] = field(default_factory=list)

    def phase_ran(self, name: str, execution_marker: str) -> bool:
        """True only if the phase's own output contains the marker."""
        return execution_marker in self.phase_text.get(name, "")

    def parity_for(self, pair: str) -> dict | None:
        for rec in self.parity:
            if rec.get("pair") == pair:
                return rec
        return None


def parse_phase_log(text: str,
                    gate_marker_re: str = r"^[A-Z0-9_]*GATE_PASS$"
                    ) -> PhaseLogEvidence:
    """Extract phase medians, raw times, parity records, and gate markers."""
    ev = PhaseLogEvidence()
    marks = [(m.start(), m.group(2)) for m in _PHASE_RE.finditer(text)]
    for i, (start, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        chunk = text[start:end]
        ev.phase_text[name] = chunk
        med = _MEDIAN_RE.search(chunk)
        if med:
            ev.phases[name] = float(med.group(1))
        times = re.search(r'"times_ms":\s*\[([^\]]*)\]', chunk)
        if times:
            ev.times_ms[name] = [float(x) for x in
                                 re.findall(r"[0-9.]+", times.group(1))]
    for m in _PARITY_RE.finditer(text):
        try:
            ev.parity.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    ev.gate_markers = re.findall(gate_marker_re, text, flags=re.MULTILINE)
    return ev


def perf_distribution(median_ms: float,
                      times_ms: list[float] | None = None) -> dict:
    xs = sorted(times_ms or [])
    if xs:
        p50 = statistics.median(xs)
        p95 = xs[min(len(xs) - 1, max(0, round(0.95 * (len(xs) - 1))))]
    else:
        p50 = p95 = median_ms
    return {"p50_ms": p50, "p95_ms": p95, "samples": len(xs)}


def quality_from_parity(rec: dict | None) -> dict:
    """Bit-exact parity record -> quality verdict; absence is a failure."""
    if rec is None:
        return {"verdict": None,
                "note": "no parity record found for the designated pair"}
    if rec.get("bitexact") is True:
        return {"verdict": "exact", "max_abs": rec.get("max_abs")}
    return {"verdict": "failed", "max_abs": rec.get("max_abs"),
            "note": "outputs are not bit-exact"}


def build_receipt(plan_id: str, backend_cap: BackendCap, *,
                  candidate_phase: str, baseline_phase: str,
                  evidence: PhaseLogEvidence, log_text: str,
                  parity_pair: str, passes: list[str],
                  authenticity: dict[str, bool],
                  require_gate_marker: bool = True,
                  **meta) -> Receipt:
    """Assemble a receipt strictly from parsed evidence.

    Missing evidence lands as promote blockers (empty perf, missing
    quality verdict); a forbidden-log hit lands in ``fallback_hits``.
    ``meta`` passes through Receipt fields (hardware, revisions, ...).
    """
    def dist(phase: str) -> dict:
        if phase not in evidence.phases:
            return {}
        return perf_distribution(evidence.phases[phase],
                                 evidence.times_ms.get(phase))

    auth = dict(authenticity)
    if require_gate_marker:
        auth["e2e_gate_marker_present"] = bool(evidence.gate_markers)
    scan = scan_log(backend_cap, log_text)
    return Receipt(
        plan_id=plan_id,
        backend=backend_cap.backend,
        backend_version=backend_cap.version,
        passes=list(passes),
        perf=dist(candidate_phase),
        baseline_perf=dist(baseline_phase),
        quality=quality_from_parity(evidence.parity_for(parity_pair)),
        authenticity=auth,
        fallback_hits=list(scan.hits),
        **meta,
    )
