"""wGraph stream contracts.

A stream is a typed, rate-aware, bounded channel between two nodes or
regions.  Streams are first-class because real-time deployments are shaped
by queue bounds, backpressure policy, and deadlines — not just by "there is
an edge".  Every stream MUST declare a bounded queue and an overflow policy;
unbounded queues make throughput benchmarks look healthy while the system
silently accumulates lag.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Modality(str, Enum):
    TOKEN = "token"
    LATENT = "latent"
    FRAME = "frame"
    AUDIO = "audio"
    ACTION = "action"
    CONTROL = "control"
    MULTIVIEW = "multiview"


class Backpressure(str, Enum):
    BLOCK = "block"            # producer waits (offline / quality-first)
    DROP_OLDEST = "drop_oldest"  # real-time display streams
    COALESCE = "coalesce"      # merge pending items (e.g. latest-obs regrounding)
    REJECT = "reject"          # refuse new item (robot action queues)


@dataclass
class StreamSpec:
    """Declarative contract for one producer→consumer channel."""

    id: str
    modality: Modality
    producer: str                # node/region id
    consumer: str                # node/region id
    chunk_size: int | None = None    # None => variable-size items
    rate_hz: float | None = None     # nominal production rate; None => on-demand
    variable_rate: bool = False
    timestamped: bool = True
    bounded_queue: int = 2
    backpressure: Backpressure = Backpressure.BLOCK
    deadline_ms: float | None = None   # per-item consumer deadline
    sync_group: str | None = None      # streams sharing a group must present in sync
    description: str = ""

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.id:
            errs.append("stream id must be non-empty")
        if not self.producer or not self.consumer:
            errs.append(f"stream '{self.id}': producer and consumer required")
        if self.producer == self.consumer:
            errs.append(f"stream '{self.id}': self-loop stream not allowed")
        if self.bounded_queue < 1:
            errs.append(f"stream '{self.id}': bounded_queue must be >= 1")
        if self.chunk_size is not None and self.chunk_size < 1:
            errs.append(f"stream '{self.id}': chunk_size must be >= 1")
        if self.rate_hz is not None and self.rate_hz <= 0:
            errs.append(f"stream '{self.id}': rate_hz must be > 0")
        if self.deadline_ms is not None and self.deadline_ms <= 0:
            errs.append(f"stream '{self.id}': deadline_ms must be > 0")
        if self.variable_rate and self.rate_hz is not None:
            errs.append(f"stream '{self.id}': variable_rate excludes fixed rate_hz")
        return errs
