"""Chunk policies on streaming edges.

A streaming producer emits one item at a time; the consumer fires only
when the accumulated buffer satisfies a *chunk policy*. The policy is
data plus a tiny decision function, so an edge can declare "decode
every 3 tokens", "sliding 4-frame window every 2 frames", or "2 new
frames plus 1 frame of left context" without either endpoint changing.

Fail-closed contracts:

* policy parameters are validated at construction (``ValueError``) —
  in particular ``stride <= window``: a stride larger than the window
  would silently skip items between consecutive windows, which is
  data loss dressed up as configuration;
* overflow of the RAW item buffer follows the declared
  :class:`wllm.graph.streams.Backpressure` semantics (block raises,
  drop_oldest / coalesce evict and count, reject refuses and counts)
  — the same meanings the executor's stream channels honor;
* ``flush`` is explicit: the tail of a stream is delivered only when
  the caller declares end-of-stream. A channel must never guess that
  a stream ended, and must never silently discard a partial tail —
  the two failure modes an implicit flush would have to pick between.

A policy instance may hold per-stream progress (sliding windows and
left context must know what was already delivered), so one policy
instance belongs to exactly one channel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..graph.streams import Backpressure


class ChunkPolicy(Protocol):
    """Decides when the buffered items form a chunk worth firing.

    ``ready`` returns ``(chunk, consume)`` — the chunk to deliver and
    how many items to drop from the buffer front — or ``None`` while
    the buffer does not yet satisfy the policy. ``consume`` may be
    smaller than the chunk (retained items provide context for later
    chunks) but must be >= 1: a firing that consumes nothing could
    fire forever. ``tail`` renders the remaining buffer as one final
    partial chunk (or ``None`` if every item was already delivered);
    it is called only by an explicit ``flush``.
    """

    def ready(self, buffered: list) -> tuple[list, int] | None: ...

    def tail(self, buffered: list) -> list | None: ...


@dataclass
class FixedChunk:
    """Fire every ``size`` items; no overlap, no context."""

    size: int

    def __post_init__(self):
        if self.size < 1:
            raise ValueError(f"FixedChunk size must be >= 1, "
                             f"got {self.size}")

    def ready(self, buffered: list) -> tuple[list, int] | None:
        if len(buffered) >= self.size:
            return list(buffered[:self.size]), self.size
        return None

    def tail(self, buffered: list) -> list | None:
        return list(buffered) if buffered else None


@dataclass
class SlidingWindow:
    """Fire ``[0:window]``, consume ``stride``.

    After a fire the buffer retains the last ``window - stride``
    delivered items as context for the next window. ``stride`` must be
    <= ``window``: a larger stride would drop the items between
    consecutive windows without any counter noticing. Flush delivers a
    final partial window (retained context + undelivered items) only
    when undelivered items exist — retained context alone was already
    delivered and is never re-fired.
    """

    window: int
    stride: int
    # delivered items currently at the buffer head (context)
    _delivered_head: int = field(default=0, init=False, repr=False,
                                 compare=False)

    def __post_init__(self):
        if self.window < 1:
            raise ValueError(f"SlidingWindow window must be >= 1, "
                             f"got {self.window}")
        if self.stride < 1:
            raise ValueError(f"SlidingWindow stride must be >= 1, "
                             f"got {self.stride}")
        if self.stride > self.window:
            raise ValueError(
                f"SlidingWindow stride ({self.stride}) must be <= "
                f"window ({self.window}); a larger stride silently "
                f"skips items between windows")

    def ready(self, buffered: list) -> tuple[list, int] | None:
        if len(buffered) >= self.window:
            self._delivered_head = self.window - self.stride
            return list(buffered[:self.window]), self.stride
        return None

    def tail(self, buffered: list) -> list | None:
        head = min(self._delivered_head, len(buffered))
        if len(buffered) > head:
            self._delivered_head = 0
            return list(buffered)
        return None


@dataclass
class LeftContext:
    """Fire ``left`` items of context plus ``chunk`` new items.

    Only the new items are consumed from the buffer; the policy keeps
    the trailing ``left`` delivered items internally as the next
    chunk's context. The first chunk has no context yet (nothing was
    delivered) and is exactly the first ``chunk`` items. Flush
    prepends the current context to the undelivered tail; context
    alone (no new items) is never re-fired.
    """

    chunk: int
    left: int
    _ctx: list = field(default_factory=list, init=False, repr=False,
                       compare=False)

    def __post_init__(self):
        if self.chunk < 1:
            raise ValueError(f"LeftContext chunk must be >= 1, "
                             f"got {self.chunk}")
        if self.left < 0:
            raise ValueError(f"LeftContext left must be >= 0, "
                             f"got {self.left}")

    def ready(self, buffered: list) -> tuple[list, int] | None:
        if len(buffered) >= self.chunk:
            out = self._ctx + list(buffered[:self.chunk])
            self._ctx = out[-self.left:] if self.left else []
            return out, self.chunk
        return None

    def tail(self, buffered: list) -> list | None:
        if buffered:
            out = self._ctx + list(buffered)
            self._ctx = []
            return out
        return None


class ChunkedChannel:
    """Bounded raw-item buffer with a chunk policy on the firing side.

    ``put`` accepts one produced item and returns the list of chunks
    that became ready (zero or more). Overflow of the RAW buffer obeys
    the declared :class:`Backpressure` policy, mirroring the
    executor's stream channels. Counters: ``items_in`` accepted items,
    ``chunks_out`` fired chunks (flush included), ``dropped``
    evictions, ``rejected`` refusals.

    If ``capacity`` is smaller than the policy's fill requirement the
    channel can never fire; overflowing items are then rejected or
    dropped per the declared policy and the counters expose it —
    nothing is lost silently.
    """

    def __init__(self, policy: ChunkPolicy, capacity: int,
                 backpressure: Backpressure):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.policy = policy
        self.capacity = capacity
        self.backpressure = backpressure
        self._buf: list = []
        self.items_in = 0
        self.chunks_out = 0
        self.dropped = 0
        self.rejected = 0

    def put(self, item) -> list[list]:
        """Accept one item; return the chunks this put made ready."""
        if len(self._buf) >= self.capacity:
            if self.backpressure == Backpressure.DROP_OLDEST:
                self._buf.pop(0)
                self.dropped += 1
            elif self.backpressure == Backpressure.COALESCE:
                self._buf.pop()      # newest is replaced by the item
                self.dropped += 1
            elif self.backpressure == Backpressure.REJECT:
                self.rejected += 1
                return []
            else:
                raise BufferError(
                    f"chunk buffer full (capacity={self.capacity}, "
                    f"policy=block); producer must wait — synchronous "
                    f"stream refuses to deadlock")
        self._buf.append(item)
        self.items_in += 1
        return self._drain_ready()

    def _drain_ready(self) -> list[list]:
        fired: list[list] = []
        while True:
            hit = self.policy.ready(list(self._buf))
            if hit is None:
                return fired
            chunk, consume = hit
            ok = (isinstance(consume, int)
                  and 1 <= consume <= len(self._buf))
            if not ok:
                raise RuntimeError(
                    f"chunk policy consumed {consume!r} of "
                    f"{len(self._buf)} buffered items; consume must "
                    f"be an int in [1, buffered] — refusing to guess")
            del self._buf[:consume]
            self.chunks_out += 1
            fired.append(chunk)

    def flush(self) -> list | None:
        """Deliver the undelivered tail as one final partial chunk.

        Explicit by design: only the caller knows the stream ended;
        an implicit flush would either guess an end (splitting live
        streams mid-flight) or silently drop the tail. Returns None
        when every buffered item was already delivered.
        """
        chunk = self.policy.tail(list(self._buf))
        self._buf.clear()
        if chunk is None:
            return None
        self.chunks_out += 1
        return chunk
