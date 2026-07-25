"""Optimization trace store: append-only experiment memory.

The trace store is wLLM's shared memory of what has been tried:
successes AND failures both persist, so no planner or agent ever
re-explores a known-dead configuration blind. A trace pins one
candidate configuration on one (model, hardware, runtime, workload)
key together with its outcome; for rejected/failed outcomes a reason
is mandatory, because the reason is exactly what a future planner
needs in order to skip that config with cause.

Constraints (stdlib only):
- JSONL on disk, one trace per line, append-only; nothing is ever
  rewritten or deleted.
- ``recorded`` is an explicit "YYYY-MM-DD" string, never the ambient
  clock — seeds and replays must be reproducible byte-for-byte.
- Corrupt JSONL lines are skipped at load and counted in
  ``TraceStore.corrupt_lines``: loading never crashes the store, and
  never silently pretends the file was fully readable.
- Appends fail closed: an invalid trace raises ``ValueError`` with
  every validation problem spelled out, and writes nothing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

STATUSES = ("accepted", "rejected", "failed")
_NEEDS_REASON = ("rejected", "failed")
# "recorded" is part of identity: a re-measurement on a later date is a
# NEW trace (drift stays visible in history), while re-running the same
# seed with its fixed date stays idempotent.
ID_FIELDS = ("model", "hardware", "runtime", "workload", "candidate",
             "status", "recorded")


@dataclass
class Trace:
    """One recorded optimization outcome (success or failure)."""

    model: str
    hardware: str        # e.g. "1xH200", "2xH200"
    runtime: str         # e.g. "wllm-serving", "wllm-native"
    workload: str        # human-readable workload key
    candidate: dict      # config knobs, e.g. {"pass": ..., "gpus": 2}
    status: str          # accepted | rejected | failed
    reason: str = ""     # REQUIRED non-empty for rejected/failed
    metrics: dict = field(default_factory=dict)  # e.g. {"speedup": 1.44}
    evidence: str = ""   # job id / receipt path / report pointer
    recorded: str = ""   # explicit "YYYY-MM-DD", never ambient clock

    # ------------------------------------------------------------- id
    @property
    def trace_id(self) -> str:
        """Content id over identity fields (not metrics/evidence).

        Same outcome, same config, same recorded date dedups even if
        prose or metric detail differs — seeding stays idempotent. A
        changed knob, model, hardware, runtime, workload, status, or a
        LATER recorded date is a new trace: re-measurements append, so
        outcome drift (a speedup that stops reproducing) stays visible
        in history instead of being swallowed by dedup. Candidate key
        order is normalized (sort_keys).
        """
        basis = {k: getattr(self, k) for k in ID_FIELDS}
        blob = json.dumps(basis, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    # ----------------------------------------------------- validation
    def validate(self) -> list[str]:
        """Empty list == storable. Anything else must fail closed."""
        errs: list[str] = []
        if self.status not in STATUSES:
            errs.append(f"unknown status {self.status!r} "
                        f"(want one of {list(STATUSES)})")
        if self.status in _NEEDS_REASON and not str(self.reason).strip():
            errs.append(f"status {self.status!r} requires a non-empty "
                        f"reason (the reason is what lets a planner "
                        f"skip this config with cause)")
        if not str(self.model).strip():
            errs.append("model is empty")
        if not isinstance(self.candidate, dict) or not self.candidate:
            errs.append("candidate is empty (a trace must pin the "
                        "config it tested)")
        try:
            datetime.strptime(self.recorded, "%Y-%m-%d")
        except (TypeError, ValueError):
            errs.append(f"recorded {self.recorded!r} is not an explicit "
                        f"YYYY-MM-DD date (ambient clock is banned)")
        return errs

    # ----------------------------------------------------- round-trip
    def to_dict(self) -> dict:
        d = asdict(self)
        d["trace_id"] = self.trace_id  # stored for grep; derived value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Trace":
        d = dict(d)
        d.pop("trace_id", None)        # derived; recomputed on demand
        return cls(**d)


class TraceStore:
    """Append-only JSONL trace store.

    Loading tolerates damage: any line that is not valid JSON or not a
    valid :class:`Trace` schema is skipped and counted in
    ``corrupt_lines`` — the store never crashes on a bad line, and
    ``corrupt_lines`` is the honest record that the file was not fully
    readable (never silently zeroed by an otherwise-successful load).
    Appending an already-known ``trace_id`` is skipped and counted in
    ``deduped``, which makes seeding idempotent.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.corrupt_lines = 0
        self.deduped = 0
        self._traces: list[Trace] = []
        self._ids: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
                if not isinstance(doc, dict):
                    raise TypeError("trace line is not a JSON object")
                trace = Trace.from_dict(doc)
            except (json.JSONDecodeError, TypeError):
                self.corrupt_lines += 1
                continue
            if trace.validate():
                # schema-shaped but semantically invalid (bad status,
                # missing reason, ...): counted, never served to queries
                self.corrupt_lines += 1
                continue
            self._traces.append(trace)
            self._ids.add(trace.trace_id)

    # ---------------------------------------------------------- write
    def append(self, trace: Trace) -> str:
        """Validate fail-closed, dedup by trace_id, append one line."""
        errs = trace.validate()
        if errs:
            raise ValueError("trace refused (fail closed): "
                             + "; ".join(errs))
        tid = trace.trace_id
        if tid in self._ids:
            self.deduped += 1
            return tid
        # defensive copy: the caller keeps its object; later mutation of
        # the caller's dicts must not desynchronize memory, ids, and disk
        stored = Trace.from_dict(trace.to_dict())
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(stored.to_dict(), sort_keys=True) + "\n")
        self._traces.append(stored)
        self._ids.add(tid)
        return tid

    # ----------------------------------------------------------- read
    def all(self) -> list[Trace]:
        return list(self._traces)

    def query(self, model: str | None = None,
              hardware: str | None = None,
              runtime: str | None = None,
              status: str | None = None,
              pass_name: str | None = None) -> list[Trace]:
        """Filter traces; ``pass_name`` matches candidate.get("pass")."""
        out: list[Trace] = []
        for t in self._traces:
            if model is not None and t.model != model:
                continue
            if hardware is not None and t.hardware != hardware:
                continue
            if runtime is not None and t.runtime != runtime:
                continue
            if status is not None and t.status != status:
                continue
            if pass_name is not None \
                    and t.candidate.get("pass") != pass_name:
                continue
            out.append(t)
        return out

    def failure_patterns(self) -> dict[str, list[str]]:
        """Pass name -> sorted unique reasons from rejected/failed."""
        pat: dict[str, set[str]] = {}
        for t in self._traces:
            if t.status not in _NEEDS_REASON:
                continue
            name = str(t.candidate.get("pass", "<no-pass>"))
            pat.setdefault(name, set()).add(t.reason)
        return {k: sorted(v) for k, v in sorted(pat.items())}

    def known_bad(self, model: str, hardware: str,
                  candidate: dict) -> Trace | None:
        """Latest matching trace, returned only if it is bad.

        Matching is *subset* semantics: every key in ``candidate`` must
        be present in the trace's candidate with an equal value; the
        trace may carry extra detail keys (e.g. a compile mode). This
        is what lets a planner probe with ``{"pass": ..., "gpus": ...}``
        and still hit richer recorded configs.

        Rehabilitation: the LATEST matching trace (append order) wins —
        if a config was rejected once and later re-measured as
        accepted, this returns ``None``; a rejection is evidence, not a
        life sentence.
        """
        for t in reversed(self._traces):
            if t.model != model or t.hardware != hardware:
                continue
            if not isinstance(t.candidate, dict):
                continue
            if all(t.candidate.get(k) == v for k, v in candidate.items()):
                return t if t.status in _NEEDS_REASON else None
        return None


# --------------------------------------------------------------- seeds
def beta_seed_traces() -> list[Trace]:
    """The six measured Alpha/Beta outcomes, verbatim from the reports.

    Every number below is copied from docs/ALPHA_REPORT.md or
    docs/BETA_REPORT.md (each citing its real SLURM job); nothing here
    is invented. ``recorded`` is the date the outcome entered the
    ledger: Alpha-report rows on 2026-07-24 (report day), the two
    job-196293 E2E rows on 2026-07-25 (digested from the Beta Day-1
    report; the job itself ran the evening of 2026-07-24). The OpenVLA
    runtime attribution follows registry_data/wllm_native.yaml, which
    carries the same measured note.
    """
    wan = "Wan-AI/Wan2.2-TI2V-5B"
    wan_load = "t2v E2E 33f@480x832, 20 denoise steps"
    return [
        # docs/BETA_REPORT.md scorecard row 1 + milestones 8/10;
        # job 196293 (logs/wllm_wan22_cfgpar_e2e_r3_196293.out).
        Trace(model=wan, hardware="2xH200", runtime="wllm-serving",
              workload=wan_load,
              candidate={"pass": "cfg_branch_parallel", "gpus": 2},
              status="accepted",
              reason="frame-level bit-exact vs sequential reference; "
                     "one CFG branch per GPU, no cross-rank reductions",
              metrics={"speedup": 1.44, "baseline_ms": 5762.0,
                       "denoise_speedup": 1.74},
              evidence="job 196293 "
                       "(logs/wllm_wan22_cfgpar_e2e_r3_196293.out); "
                       "docs/BETA_REPORT.md scorecard + milestones 8/10",
              recorded="2026-07-25"),
        # docs/ALPHA_REPORT.md scorecard row 1 + gate note 1 (jobs
        # 195301 -> 195356 -> 195374); docs/BETA_REPORT.md law 1.
        Trace(model=wan, hardware="1xH200", runtime="wllm-serving",
              workload=wan_load,
              candidate={"pass": "torch_compile_max_autotune",
                         "mode": "max-autotune-no-cudagraphs",
                         "gpus": 1},
              status="rejected",
              reason="refused by the exact gate: bf16 fusion "
                     "reordering across 20 denoise steps drifts the "
                     "trajectory (frame drift mean 7.4/255); "
                     "promotable only under a video-quality contract",
              metrics={"speedup": 1.43, "baseline_ms": 5615.0,
                       "candidate_ms": 4021.0,
                       "frame_drift_mean_255": 7.4},
              evidence="jobs 195301/195356/195374; "
                       "docs/ALPHA_REPORT.md scorecard; "
                       "docs/BETA_REPORT.md verifier law 1",
              recorded="2026-07-24"),
        # docs/BETA_REPORT.md verifier law 4, quantified E2E in job
        # 196293 (refusal path of scripts/receipt_wan22_cfgpar.py).
        Trace(model=wan, hardware="1xH200", runtime="wllm-serving",
              workload=wan_load,
              candidate={"pass": "cfg_batched", "gpus": 1},
              status="rejected",
              reason="not bit-exact: pipeline-native batched CFG "
                     "differs from sequential branches by up to "
                     "251/255 per pixel (visibly different video)",
              metrics={"max_abs_255": 251},
              evidence="job 196293 "
                       "(logs/wllm_wan22_cfgpar_e2e_r3_196293.out); "
                       "docs/BETA_REPORT.md verifier law 4",
              recorded="2026-07-25"),
        # docs/ALPHA_REPORT.md gate note 1: reduce-overhead cudagraphs
        # measured slower (7.1 s vs 5615 ms baseline) because of
        # re-recording; negative result retained on purpose.
        Trace(model=wan, hardware="1xH200", runtime="wllm-serving",
              workload=wan_load,
              candidate={"pass": "torch_compile_reduce_overhead",
                         "mode": "reduce-overhead", "gpus": 1},
              status="rejected",
              reason="slower than baseline: cudagraph re-recording "
                     "dominates (measured 7.1 s vs 5615 ms baseline)",
              metrics={"candidate_ms": 7100.0, "baseline_ms": 5615.0},
              evidence="docs/ALPHA_REPORT.md gate note 1 "
                       "(job family 195301-195374)",
              recorded="2026-07-24"),
        # docs/ALPHA_REPORT.md scorecard row 2 (job 195433 + tie
        # probe); docs/BETA_REPORT.md scorecard + verifier law 2.
        Trace(model="Qwen/Qwen3-VL-8B-Instruct", hardware="1xH200",
              runtime="wllm-serving",
              workload="AR decode, 128 new tokens",
              candidate={"pass": "static_kv_cache", "gpus": 1},
              status="accepted",
              reason="tie-aware exact: the single greedy flip sits at "
                     "a proven argmax tie (top-2 logit gap 0.0) — "
                     "arbitration, not divergence",
              metrics={"speedup": 2.75, "baseline_ms": 2668.0,
                       "candidate_ms": 968.0},
              evidence="job 195433 + tie probe; docs/ALPHA_REPORT.md "
                       "scorecard; docs/BETA_REPORT.md verifier law 2",
              recorded="2026-07-24"),
        # docs/ALPHA_REPORT.md scorecard row 3 (jobs 195638 -> r2 ->
        # 195701); docs/BETA_REPORT.md scorecard + verifier law 3.
        Trace(model="openvla/openvla-7b", hardware="1xH200",
              runtime="wllm-native",
              workload="single-action prediction",
              candidate={"pass": "native_bf16", "gpus": 1},
              status="accepted",
              reason="native-precision restoration: the checkpoint "
                     "declares bf16 (torch_dtype bfloat16, 13 GB / 7B "
                     "params); the naive fp32 load is an upcast "
                     "variant, not the oracle",
              metrics={"speedup": 4.59, "baseline_ms": 133.0,
                       "candidate_ms": 29.0},
              evidence="jobs 195638 -> 195701; docs/ALPHA_REPORT.md "
                       "scorecard; docs/BETA_REPORT.md verifier law 3",
              recorded="2026-07-24"),
    ]


def seed_beta_traces(store: TraceStore) -> int:
    """Seed the measured Alpha/Beta outcomes; return newly-added count.

    Idempotent: traces already present (by ``trace_id``) go through
    the store's dedup path, so running the seed twice adds nothing.
    """
    before = len(store.all())
    for trace in beta_seed_traces():
        store.append(trace)
    return len(store.all()) - before
