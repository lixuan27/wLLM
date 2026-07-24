"""Scheduler base: request records and step accounting.

A stage scheduler owns admission and step batching for one stage; the
model executes, the scheduler decides *what runs together*. Stats are
authenticity evidence: `max_step_batch >= 2` proves continuous batching
actually happened, `steps` proves work went through this scheduler and
not around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...sampling import read_param


@dataclass
class ScheduledRequest:
    request_id: str
    prompt_token_ids: list[int]
    params: Any = None
    context: dict = field(default_factory=dict)
    output_token_ids: list[int] = field(default_factory=list)
    finished: bool = False
    finish_reason: str | None = None

    @property
    def max_tokens(self) -> int:
        return int(read_param(self.params, "max_tokens", 16))

    @property
    def stop_token_ids(self) -> tuple:
        return tuple(read_param(self.params, "stop_token_ids", ()) or ())


class SchedulerStats:
    def __init__(self):
        self.steps = 0
        self.max_step_batch = 0
        self.admitted = 0
        self.completed = 0

    def record_step(self, batch_size: int) -> None:
        self.steps += 1
        self.max_step_batch = max(self.max_step_batch, batch_size)

    def as_dict(self) -> dict:
        return {"steps": self.steps, "max_step_batch": self.max_step_batch,
                "admitted": self.admitted, "completed": self.completed}


class BaseScheduler:
    def __init__(self, max_num_seqs: int = 64):
        if max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        self.max_num_seqs = max_num_seqs
        self.waiting: list[ScheduledRequest] = []
        self.running: list[ScheduledRequest] = []
        self.stats = SchedulerStats()

    def add(self, req: ScheduledRequest) -> None:
        known = {r.request_id for r in self.waiting + self.running}
        if req.request_id in known:
            raise ValueError(f"duplicate request id {req.request_id!r}")
        self.waiting.append(req)
        self.stats.admitted += 1

    def abort(self, request_id: str) -> bool:
        """Remove a request wherever it sits; True if it was present.

        A cancelled or crashed ``generate`` must never leave its request
        behind — a poisoned resident would re-crash every later batch and
        blame innocent requests.
        """
        for queue in (self.waiting, self.running):
            for r in queue:
                if r.request_id == request_id:
                    queue.remove(r)
                    return True
        return False

    def _fail_batch(self, batch: list[ScheduledRequest], exc: Exception) -> None:
        """Mark a crashed batch failed and retire it before re-raising."""
        for r in batch:
            r.finished = True
            r.finish_reason = "error"
            r.context["error"] = f"{type(exc).__name__}: {exc}"
        self._retire_finished()

    def _admit(self) -> None:
        while self.waiting and len(self.running) < self.max_num_seqs:
            self.running.append(self.waiting.pop(0))

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def _retire_finished(self) -> None:
        done = [r for r in self.running if r.finished]
        self.stats.completed += len(done)
        self.running = [r for r in self.running if not r.finished]
