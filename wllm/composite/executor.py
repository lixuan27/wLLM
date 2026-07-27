"""Walk execution: session isolation, placement, loops, streams.

The executor binds component ids to callables, threads a context dict
through the walk, and enforces the contracts the IR declares:

* session states live in a :class:`SessionStore` — reads/writes are
  keyed by (session, state) so two sessions can never see each other's
  KV, world history, or action history; ``reset`` provably clears.
* placement is data, not code: a plan maps component -> device label,
  and every invocation is recorded with its device so tests (and later
  profilers) can assert where work actually ran.
* streaming edges honor the edge's bounded queue and backpressure
  policy (block / drop_oldest / coalesce / reject).

Component callables have the signature ``fn(ctx, state) -> dict`` where
``ctx`` is the walk context (merged into on return) and ``state`` is a
mutable per-(session, component) mapping for its owned states.

Concurrency. One executor may run walks for different sessions on
different threads at the same time — that is the point of per-session
state. What that costs is bounded: :class:`SessionStore` guards its map
with a lock, ``channel`` keys include the session, and ``invocations``
is only appended to. Two threads driving the SAME session concurrently
is NOT supported (they would interleave writes into one state dict);
callers serialize per session, which a request-per-session scheduler
does for free. The session id of the walk running on the current
thread is available to component code via :func:`current_session`.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from ..graph.streams import Backpressure
from .graph import ComponentGraph
from .walk import Loop, Par, Seq, Stream, Walk

_CURRENT = threading.local()


def current_session() -> str:
    """Session id of the walk running on this thread.

    A component receives ``(ctx, state)``: ctx is walk data and state is
    already scoped for it, so neither carries the request identity. A
    component that must key an EXTERNAL resource by request — a batching
    gate, a device slot, a KV pool — needs that identity anyway. It is
    exposed as thread-local runtime context rather than as a ctx key so
    that it cannot be written by a component, cannot leak into a walk's
    outputs, and cannot be mistaken for model data.

    Raises ``RuntimeError`` outside a walk: there is no default session,
    and guessing one would silently cross-wire two requests.
    """
    session = getattr(_CURRENT, "session", None)
    if session is None:
        raise RuntimeError(
            "current_session() called outside a walk; the executor binds "
            "it only for the duration of WalkExecutor.run()")
    return session


class SessionStore:
    """Per-session state with hard isolation and provable reset.

    Reset applies to quiescent sessions: resetting while a walk for the
    same session is in flight leaves that walk holding orphaned state
    dicts (its writes are discarded, not merged). Callers must serialize
    reset against in-flight walks.
    """

    def __init__(self):
        self._data: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def state(self, session: str, component: str) -> dict[str, Any]:
        with self._lock:
            return self._data.setdefault(session, {}).setdefault(component, {})

    def reset(self, session: str) -> None:
        with self._lock:
            self._data.pop(session, None)

    def sessions(self) -> list[str]:
        with self._lock:
            return sorted(self._data)


class BoundedChannel:
    """In-process stream edge honoring the declared backpressure policy."""

    def __init__(self, capacity: int, policy: Backpressure):
        self.capacity = capacity
        self.policy = policy
        self._items: deque = deque()
        self.dropped = 0
        self.rejected = 0

    def put(self, item) -> bool:
        if len(self._items) < self.capacity:
            self._items.append(item)
            return True
        if self.policy == Backpressure.DROP_OLDEST:
            self._items.popleft()
            self.dropped += 1
            self._items.append(item)
            return True
        if self.policy == Backpressure.COALESCE:
            self._items[-1] = item
            self.dropped += 1
            return True
        if self.policy == Backpressure.REJECT:
            self.rejected += 1
            return False
        raise BufferError(
            f"stream queue full (capacity={self.capacity}, policy=block); "
            f"producer must wait — synchronous walk refuses to deadlock")

    def drain(self) -> list:
        out = list(self._items)
        self._items.clear()
        return out


@dataclass
class Invocation:
    component: str
    device: str
    session: str


@dataclass
class WalkExecutor:
    graph: ComponentGraph
    impls: dict[str, Callable[[dict, dict], dict]]
    placement: dict[str, str] = field(default_factory=dict)  # component -> device
    store: SessionStore = field(default_factory=SessionStore)

    def __post_init__(self):
        errs = self.graph.validate()
        if errs:
            raise ValueError(f"invalid graph: {errs}")
        missing = {c.id for c in self.graph.components} - set(self.impls)
        if missing:
            raise ValueError(f"components without implementations: "
                             f"{sorted(missing)}")
        self.invocations: list[Invocation] = []
        self._stream_edges: dict[tuple[str, str], object] = {
            (e.source, e.target): e.stream
            for e in self.graph.edges if e.stream is not None}
        # channels are per-session: stream items are session data and must
        # never leak between sessions or survive a session reset
        self.channels: dict[tuple[str, str, str], BoundedChannel] = {}

    def channel(self, session: str, source: str, target: str) -> BoundedChannel:
        key = (session, source, target)
        if key not in self.channels:
            spec = self._stream_edges[(source, target)]
            self.channels[key] = BoundedChannel(spec.bounded_queue,
                                                spec.backpressure)
        return self.channels[key]

    def reset_session(self, session: str) -> None:
        """Clear a session's states AND its stream channels."""
        self.store.reset(session)
        for key in [k for k in self.channels if k[0] == session]:
            del self.channels[key]

    # ------------------------------------------------------------------ run
    def run(self, walk: Walk, session: str, ctx: dict | None = None) -> dict:
        errs = walk.validate(self.graph)
        if errs:
            raise ValueError(f"invalid walk: {errs}")
        ctx = dict(ctx or {})
        # bind the running session for component code; restore rather than
        # clear on exit so a nested run() cannot orphan its caller's binding
        previous = getattr(_CURRENT, "session", None)
        _CURRENT.session = session
        try:
            for step in walk.steps:
                ctx = self._step(step, session, ctx)
        finally:
            _CURRENT.session = previous
        return ctx

    def _step(self, step, session: str, ctx: dict) -> dict:
        if isinstance(step, Seq):
            return self._call(step.component, session, ctx)
        if isinstance(step, Par):
            if step.join != "merge":
                raise NotImplementedError(
                    f"unknown Par join mode {step.join!r}; only 'merge' "
                    f"is implemented — refusing to guess semantics")
            merged = dict(ctx)
            seen_new: dict[str, str] = {}
            for i, branch in enumerate(step.branches):
                out = dict(ctx)
                for s in branch.steps:
                    out = self._step(s, session, out)
                for k, v in out.items():
                    if k in ctx and ctx[k] is v:
                        continue
                    if k in seen_new:
                        raise ValueError(
                            f"parallel branches both produced key {k!r} "
                            f"(branches {seen_new[k]} and {i}); joins must "
                            f"be disjoint")
                    seen_new[k] = str(i)
                    merged[k] = v
            return merged
        if isinstance(step, Loop):
            probs = step.validate()
            if probs:
                raise ValueError(f"invalid loop: {probs}")
            outer_index = ctx.get("loop_index")
            if step.until is not None:
                # a stale flag from an earlier loop must not end this one
                ctx = dict(ctx)
                ctx.pop(step.until, None)
            n = 0
            limit = (step.iterations if step.iterations is not None
                     else step.max_iterations)
            while n < limit:
                ctx["loop_index"] = n
                for s in step.body.steps:
                    ctx = self._step(s, session, ctx)
                n += 1
                if step.until is not None and ctx.get(step.until):
                    break
            else:
                if step.until is not None:
                    raise RuntimeError(
                        f"loop hit max_iterations={step.max_iterations} "
                        f"without {step.until!r} becoming true")
            ctx["loop_iterations_run"] = n
            # restore the enclosing loop's index for its remaining steps
            if outer_index is not None:
                ctx["loop_index"] = outer_index
            else:
                ctx.pop("loop_index", None)
            return ctx
        if isinstance(step, Stream):
            chan = self.channel(session, step.source, step.target)
            accepted = chan.put(ctx.get("stream_item"))
            ctx = dict(ctx)   # never mutate an inherited list in place
            ctx["stream_accepted"] = list(ctx.get("stream_accepted", []))
            ctx["stream_accepted"].append(accepted)
            return ctx
        raise TypeError(f"unknown walk step {type(step).__name__}")

    def _call(self, component: str, session: str, ctx: dict) -> dict:
        comp = self.graph.component(component)
        device = self.placement.get(component, "cpu")
        if comp.placement_domain.startswith("fixed:"):
            pinned = comp.placement_domain.split(":", 1)[1]
            if device != pinned:
                raise ValueError(
                    f"component {component!r} is pinned to {pinned!r} but "
                    f"the plan placed it on {device!r}")
        self.invocations.append(Invocation(component, device, session))
        state = self.store.state(session, component)
        out = self.impls[component](dict(ctx), state)
        if out is None:
            out = {}
        if not isinstance(out, dict):
            raise TypeError(f"component {component!r} must return a dict, "
                            f"got {type(out).__name__}")
        merged = dict(ctx)
        merged.update(out)
        return merged

    # ------------------------------------------------------------ evidence
    def devices_used(self) -> dict[str, set[str]]:
        used: dict[str, set[str]] = {}
        for inv in self.invocations:
            used.setdefault(inv.component, set()).add(inv.device)
        return used
