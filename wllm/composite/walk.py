"""Walk algebra: what one request does on a component graph.

A walk is a small program over components:

    Seq("text_encoder")                       run one component
    Par(Walk(...), Walk(...))                 branches with a disjoint join
    Loop(body, iterations=35, carry="latent") iterative refinement
    Loop(body, until="stop", carry="tokens", max_iterations=4096)
    Stream("dit", "vae")                      hand off via the edge's stream

Different request kinds (image / video / audio / action) are different
walks over the *same* graph — the graph never gets duplicated per task.

Par semantics: branches execute sequentially over shared session state
(state writes by branch 1 are visible to branch 2); only their *context*
outputs are isolated and merged, and the join requires disjoint new
keys. Components must return new values rather than mutating inherited
context values in place — in-place mutation of a shared object cannot
be detected by the join check.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Seq:
    component: str


@dataclass
class Par:
    branches: list["Walk"]
    join: str = "merge"      # merge dict outputs; first key wins is an error


@dataclass
class Loop:
    body: "Walk"
    carry: str                       # ctx key threaded through iterations
    iterations: int | None = None    # fixed count (diffusion) ...
    until: str | None = None         # ... or ctx key that turns truthy (AR)
    max_iterations: int = 4096

    def validate(self) -> list[str]:
        errs = []
        if (self.iterations is None) == (self.until is None):
            errs.append("loop needs exactly one of iterations/until")
        if self.iterations is not None and self.iterations < 1:
            errs.append("loop iterations must be >= 1")
        if self.max_iterations < 1:
            errs.append("loop max_iterations must be >= 1")
        if (self.iterations is not None
                and self.iterations > self.max_iterations):
            errs.append("loop iterations exceeds max_iterations")
        return errs


@dataclass
class Stream:
    source: str
    target: str


Step = Seq | Par | Loop | Stream


@dataclass
class Walk:
    steps: list[Step] = field(default_factory=list)

    def components(self) -> set[str]:
        out: set[str] = set()
        for s in self.steps:
            if isinstance(s, Seq):
                out.add(s.component)
            elif isinstance(s, Par):
                for b in s.branches:
                    out |= b.components()
            elif isinstance(s, Loop):
                out |= s.body.components()
            elif isinstance(s, Stream):
                out |= {s.source, s.target}
        return out

    def validate(self, graph) -> list[str]:
        """Check the walk against a ComponentGraph; empty == valid."""
        errs: list[str] = []
        known = {c.id for c in graph.components}
        missing = sorted(self.components() - known)
        if missing:
            errs.append(f"walk references unknown components: {missing}")
        stream_edges = {(e.source, e.target)
                        for e in graph.edges if e.stream is not None}
        for s in self._iter_steps():
            if isinstance(s, Loop):
                errs.extend(s.validate())
            elif isinstance(s, Stream) and (s.source, s.target) not in stream_edges:
                errs.append(f"walk streams {s.source}->{s.target} but the "
                            f"graph declares no stream edge there")
        return errs

    def _iter_steps(self):
        stack: list[Step] = list(self.steps)
        while stack:
            s = stack.pop()
            yield s
            if isinstance(s, Par):
                for b in s.branches:
                    stack.extend(b.steps)
            elif isinstance(s, Loop):
                stack.extend(s.body.steps)
