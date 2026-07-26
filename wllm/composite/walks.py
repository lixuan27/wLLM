"""Named walk sets + per-request walk state machines.

A composite model is a component graph plus a finite set of *named*
walks (:mod:`wllm.composite.walk`). One request is a SERIES of walks:
an author-provided state machine looks at the request's intent and at
the context produced so far, and names the next walk to run — so the
scheduler executes only the components each request actually needs.
``WalkExecutor.invocations`` plus ``RequestResult.walk_trail`` are the
evidence for that minimum-components property.

Contracts, all fail-closed:

* a chooser returning an unknown walk name is a ``KeyError`` naming
  the known walks — a typo must not fall back to "run something";
* a chooser that never returns ``None`` hits ``max_walks`` and raises
  ``RuntimeError`` — requests must provably terminate;
* the request context (the ``ctx`` argument) is read-only routing
  intent: it is passed (as a copy) to the chooser only and is never
  merged into the execution context — walks see only what earlier
  walks produced, so routing hints cannot leak into model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from .executor import WalkExecutor
from .walk import Walk


@dataclass
class WalkSet:
    """A model's finite, named repertoire of walks over one graph."""

    walks: dict[str, Walk] = field(default_factory=dict)

    def validate(self, graph) -> list[str]:
        """Problems with this set against a graph; empty == valid.

        An empty set is rejected (a model that names no walks can
        serve no request), as are empty / non-string names — routing
        happens by name, so every name must be usable as one.
        """
        errs: list[str] = []
        if not self.walks:
            errs.append("walk set is empty; a model must name at "
                        "least one walk")
        for name, walk in self.walks.items():
            if not isinstance(name, str) or not name:
                errs.append(f"walk name {name!r} invalid; names must "
                            f"be non-empty strings")
                continue
            if not isinstance(walk, Walk):
                errs.append(f"walk {name!r} is not a Walk "
                            f"({type(walk).__name__})")
                continue
            errs.extend(f"walk {name!r}: {e}"
                        for e in walk.validate(graph))
        return errs


class WalkStateMachine(Protocol):
    """Chooser for the next walk of a request.

    Called as ``next_walk(request_ctx, last_walk, last_output)`` where
    ``last_walk`` is ``None`` on the first call and ``last_output`` is
    the accumulated execution context so far (empty on the first
    call). Return the name of the next walk, or ``None`` to end the
    request. Both dicts are shallow copies: choosers observe routing
    intent and produced context, they do not write either.
    """

    def __call__(self, request_ctx: dict, last_walk: str | None,
                 last_output: dict) -> str | None: ...


@dataclass
class RequestResult:
    """What one request did: its products, its route, its intent.

    ``outputs``    final execution context — it starts empty, so every
                   key here was produced by some walk of this request
    ``walk_trail`` walk names in execution order (evidence that the
                   scheduler ran only what the request needed)
    ``ctx``        snapshot of the request context that drove routing;
                   by contract never merged into ``outputs``
    """

    outputs: dict
    walk_trail: list[str]
    ctx: dict


def run_request(executor: WalkExecutor, walkset: WalkSet,
                next_walk: Callable[[dict, str | None, dict],
                                    str | None],
                session: str, ctx: dict | None = None,
                max_walks: int = 64) -> RequestResult:
    """Drive one request as a chooser-directed series of walks.

    ``ctx`` is the request context (read-only intent, e.g. the task
    kind); the execution context starts empty and each walk's produced
    context is threaded into the next via ``executor.run``. The
    chooser sees ``(request_ctx, last_walk, last_output)`` and returns
    the next walk name or ``None`` to finish.
    """
    if max_walks < 1:
        raise ValueError(f"max_walks must be >= 1, got {max_walks}")
    if not walkset.walks:
        raise ValueError("walk set is empty; a request cannot choose "
                         "any walk")
    request_ctx = dict(ctx or {})
    exec_ctx: dict = {}
    trail: list[str] = []
    last_walk: str | None = None
    while True:
        name = next_walk(dict(request_ctx), last_walk, dict(exec_ctx))
        if name is None:
            break
        if not isinstance(name, str):
            raise TypeError(f"state machine must return a walk name "
                            f"or None, got {type(name).__name__}")
        if name not in walkset.walks:
            raise KeyError(
                f"state machine chose unknown walk {name!r}; known "
                f"walks: {sorted(walkset.walks)}")
        if len(trail) >= max_walks:
            raise RuntimeError(
                f"state machine did not terminate: {max_walks} walks "
                f"run without the chooser returning None (trail tail: "
                f"{trail[-6:]})")
        exec_ctx = executor.run(walkset.walks[name], session, exec_ctx)
        trail.append(name)
        last_walk = name
    return RequestResult(outputs=exec_ctx, walk_trail=trail,
                         ctx=request_ctx)
