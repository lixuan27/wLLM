"""Rendezvous: K concurrent walks -> one batched step call.

:mod:`wllm.composite.batching` answers "given a list of step requests,
how do I fuse them safely". It does not answer "where does that list
come from" — in a real deployment each request is a separate walk on a
separate thread, and nobody hands the batcher a list. The gate is that
missing half: a component implementation calls :meth:`StepGate.submit`
and blocks; when the round is full (or its wait window expires) one
caller becomes the round leader, runs the pending requests through the
:class:`~wllm.composite.batching.StepBatcher`, and hands every caller
back *only* the result keyed by its own request id.

Contracts, all fail-closed:

* a caller only ever receives ``results[its own request id]``; a
  missing key is a ``RuntimeError``, never another request's value;
* one request may have at most one step in flight — a second submit
  under the same id is a ``ValueError``, because the gate would
  otherwise have no way to route two results back to one caller;
* if the batched function raises, EVERY member of that round gets the
  exception. A round is a scheduling artifact, so its blast radius is
  the round; silently completing the survivors would hand callers
  results produced by a call that failed;
* a caller that waits past ``timeout_s`` raises ``TimeoutError`` and
  removes itself from the queue. The gate never hangs a request
  because a peer disappeared;
* ``width=1`` means "never fuse across requests" — the control arm that
  makes a batching claim falsifiable — but the authoritative evidence
  of what was actually fused is the batcher's own
  :meth:`~wllm.composite.batching.StepBatcher.max_group_size`, not the
  gate's round size;
* a round never exceeds ``width``, and anything queued beyond it stays
  for the next round with a different leader, so a busy gate rotates
  which thread submits work instead of letting one drain the fleet.

Round size is ``min(width, live)``: ``width`` caps how much latency a
request may take on for throughput, and ``live`` (participants that
joined via :meth:`participant`) means a round fires as soon as everyone
who could still join has joined, instead of waiting out the window.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .batching import StepBatcher, StepRequest


class _Ticket:
    """One in-flight step: its request, and where its result lands."""

    __slots__ = ("request", "done", "value", "error")

    def __init__(self, request: StepRequest):
        self.request = request
        self.done = False
        self.value: Any = None
        self.error: BaseException | None = None


@dataclass
class GateStats:
    """What the gate actually did — evidence, not configuration."""

    submissions: int = 0
    rounds: int = 0
    fused_rounds: int = 0      # rounds that carried more than one request
    max_round: int = 0         # biggest round the gate assembled
    partial_rounds: int = 0    # fired on the wait window, not on a full round
    live: int = 0              # participants currently joined


class StepGate:
    """Blocking rendezvous in front of a :class:`StepBatcher`.

    ``wait_s`` bounds how long a caller waits for peers before firing a
    partial round; ``timeout_s`` bounds the whole submit and turns a
    stuck round into a raised ``TimeoutError`` rather than a hang.
    """

    def __init__(self, batcher: StepBatcher, width: int,
                 wait_s: float = 5.0, timeout_s: float = 300.0,
                 poll_s: float = 0.02):
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}")
        if wait_s < 0:
            raise ValueError(f"wait_s must be >= 0, got {wait_s}")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {timeout_s}")
        if poll_s <= 0:
            raise ValueError(f"poll_s must be > 0, got {poll_s}")
        self.batcher = batcher
        self.width = width
        self.wait_s = wait_s
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self._cv = threading.Condition()
        self._queue: list[_Ticket] = []
        self._inflight: set[str] = set()
        self._firing = False
        self._partial = False
        self._live = 0
        self._stats = GateStats()

    # ------------------------------------------------------ participants
    def join(self) -> None:
        with self._cv:
            self._live += 1
            self._stats.live = self._live

    def leave(self) -> None:
        """Deregister; wakes waiters so a smaller round can fire now."""
        with self._cv:
            self._live = max(0, self._live - 1)
            self._stats.live = self._live
            self._cv.notify_all()

    @contextmanager
    def participant(self):
        self.join()
        try:
            yield self
        finally:
            self.leave()

    # ------------------------------------------------------------ submit
    def submit(self, request_id: str, component: str, payload: Any,
               signature: tuple = ()) -> Any:
        """Enqueue one step and block until this request's result is back."""
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(f"request_id must be a non-empty string, "
                             f"got {request_id!r}")
        ticket = _Ticket(StepRequest(request_id, component, payload,
                                     signature))
        start = time.monotonic()
        window = start + self.wait_s
        hard = start + self.timeout_s
        with self._cv:
            if request_id in self._inflight:
                raise ValueError(
                    f"request {request_id!r} already has a step in flight; "
                    f"the gate routes one result per request and will not "
                    f"guess which of two submits owns it")
            self._inflight.add(request_id)
            self._stats.submissions += 1
            self._queue.append(ticket)

        # Leadership and completion are separate: a round is capped at
        # `width`, so a caller can become leader for a round that does NOT
        # contain its own ticket (it queued behind one). Loop until this
        # caller's own ticket is resolved instead of assuming that leading
        # a round finished it.
        while True:
            batch: list[_Ticket] | None = None
            with self._cv:
                if ticket.done:
                    break
                if not self._firing and self._can_fire(window):
                    batch = self._take()
                else:
                    self._guard_deadline(hard, ticket, request_id)
                    self._cv.wait(timeout=self.poll_s)
            if batch is not None:
                self._run_round(batch)      # outside the lock: this is the
                                            # expensive call (a real kernel)
        with self._cv:
            self._inflight.discard(request_id)
        if ticket.error is not None:
            raise ticket.error
        return ticket.value

    # -------------------------------------------------------- internals
    def _target(self) -> int:
        """Requests a full round needs right now."""
        return min(self.width, self._live) if self._live else self.width

    def _can_fire(self, window: float) -> bool:
        return (len(self._queue) >= self._target()
                or time.monotonic() >= window)

    def _take(self) -> list[_Ticket]:
        # Cap at `width` and leave the rest queued: `width` is a promise
        # about round size, and the leftovers give the NEXT round a
        # different leader, so kernel submission rotates across threads
        # instead of one thread draining the fleet.
        batch = self._queue[:self.width]
        self._queue = self._queue[self.width:]
        self._firing = True
        self._partial = len(batch) < self._target()
        return batch

    def _guard_deadline(self, hard: float, ticket: _Ticket,
                        request_id: str) -> None:
        if time.monotonic() < hard:
            return
        self._queue = [t for t in self._queue if t is not ticket]
        self._inflight.discard(request_id)
        raise TimeoutError(
            f"step gate: request {request_id!r} waited "
            f"{self.timeout_s:.1f}s without its round completing "
            f"(width={self.width}, live={self._live}, "
            f"queued={len(self._queue)}, firing={self._firing}); "
            f"refusing to hang")

    def _run_round(self, batch: list[_Ticket]) -> None:
        error: BaseException | None = None
        results: dict[str, Any] = {}
        try:
            results = self.batcher.run([t.request for t in batch])
        except BaseException as exc:      # noqa: BLE001 — shared by the round
            error = exc
        with self._cv:
            for t in batch:
                if error is not None:
                    t.error = error
                elif t.request.request_id in results:
                    t.value = results[t.request.request_id]
                else:
                    t.error = RuntimeError(
                        f"batched run returned no result for request "
                        f"{t.request.request_id!r}; refusing to hand back "
                        f"another request's value")
                t.done = True
            self._stats.rounds += 1
            self._stats.max_round = max(self._stats.max_round, len(batch))
            if len(batch) > 1:
                self._stats.fused_rounds += 1
            if self._partial:
                self._stats.partial_rounds += 1
                self._partial = False
            self._firing = False
            self._cv.notify_all()

    # ------------------------------------------------------------ evidence
    def stats(self) -> GateStats:
        with self._cv:
            return GateStats(**vars(self._stats))
