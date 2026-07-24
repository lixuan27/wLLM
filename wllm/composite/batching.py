"""Cross-request step batching with per-request parity.

Iterative loops (denoise steps, AR decode steps) dominate composite
workloads; batching *compatible* concurrent requests at the same step
into one kernel launch is the main throughput lever. The safety rule is
parity: batching is a scheduling decision, so each request's result must
equal what it would have gotten alone — the batcher groups only requests
whose signature (component, step kind, shape) matches, and the batched
implementation is verified against the sequential one by the test suite,
never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class StepRequest:
    request_id: str
    component: str
    payload: Any
    signature: tuple = ()      # e.g. (shape, dtype, step_index)


@dataclass
class BatchRecord:
    component: str
    signature: tuple
    request_ids: list[str]
    # per-member signatures recorded independently of the grouping key so
    # cross-signature mixing is falsifiable, not true by construction
    member_signatures: list[tuple] = None


@dataclass
class StepBatcher:
    """Groups compatible step requests and runs them in one call.

    ``batched_fns[component]`` takes a list of payloads and returns a
    list of results in the same order. ``max_batch`` bounds group size
    so latency-critical requests are not starved behind giant batches.
    """

    batched_fns: dict[str, Callable[[list], list]]
    max_batch: int = 8
    records: list[BatchRecord] = field(default_factory=list)

    def run(self, requests: Sequence[StepRequest]) -> dict[str, Any]:
        if self.max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        unknown = sorted({r.component for r in requests}
                         - set(self.batched_fns))
        if unknown:
            raise KeyError(f"no batched implementation for: {unknown}")
        dupes = self._duplicate_ids(requests)
        if dupes:
            raise ValueError(f"duplicate request ids: {dupes}")

        results: dict[str, Any] = {}
        groups: dict[tuple, list[StepRequest]] = {}
        for r in requests:                       # arrival order preserved
            groups.setdefault((r.component, r.signature), []).append(r)
        for (component, signature), members in groups.items():
            for start in range(0, len(members), self.max_batch):
                chunk = members[start:start + self.max_batch]
                outs = self.batched_fns[component]([m.payload for m in chunk])
                if len(outs) != len(chunk):
                    raise RuntimeError(
                        f"batched fn for {component!r} returned {len(outs)} "
                        f"results for {len(chunk)} requests; refusing to "
                        f"guess the mapping")
                for m, out in zip(chunk, outs):
                    results[m.request_id] = out
                self.records.append(BatchRecord(
                    component=component, signature=signature,
                    request_ids=[m.request_id for m in chunk],
                    member_signatures=[m.signature for m in chunk]))
        return results

    @staticmethod
    def _duplicate_ids(requests: Sequence[StepRequest]) -> list[str]:
        seen: set[str] = set()
        dupes: list[str] = []
        for r in requests:
            if r.request_id in seen and r.request_id not in dupes:
                dupes.append(r.request_id)
            seen.add(r.request_id)
        return dupes

    # ------------------------------------------------------------ evidence
    def max_group_size(self) -> int:
        return max((len(b.request_ids) for b in self.records), default=0)

    def cross_signature_mixes(self) -> int:
        """Batches whose members' own signatures disagree.

        Computed from per-member records, independent of the grouping
        key — if grouping ever regresses, this becomes non-zero and the
        pinning test fails.
        """
        mixes = 0
        for b in self.records:
            if len(set(b.member_signatures or [b.signature])) != 1:
                mixes += 1
        return mixes
