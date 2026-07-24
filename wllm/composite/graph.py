"""Component graph: what a composite multimodal system is made of.

Components are typed by :class:`wllm.graph.regions.NodeOp` and declare
the states they own; edges declare data flow and may carry a stream
contract. Validation is structural and fail-closed: an invalid graph is
rejected with reasons before any walk runs on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..graph.regions import NodeOp
from ..graph.states import StateSpec
from ..graph.streams import StreamSpec


@dataclass
class Component:
    """One executable unit (encoder, DiT, VAE, action head, ...)."""

    id: str
    op: NodeOp
    # states this component owns (exactly one owner per state id)
    states: list[StateSpec] = field(default_factory=list)
    # execution facts the planner/batcher may rely on
    batchable: bool = False
    supports_cuda_graph: bool = False
    placement_domain: str = "gpu"        # gpu | cpu | any | fixed:<device>
    attrs: dict = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    stream: StreamSpec | None = None     # None == plain call edge


@dataclass
class ComponentGraph:
    name: str
    components: list[Component] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def component(self, cid: str) -> Component:
        for c in self.components:
            if c.id == cid:
                return c
        raise KeyError(f"unknown component {cid!r} in graph {self.name!r}")

    def state_owner(self, state_id: str) -> str | None:
        for c in self.components:
            if any(s.id == state_id for s in c.states):
                return c.id
        return None

    # ------------------------------------------------------------ validation
    def validate(self) -> list[str]:
        """Structural problems; empty list == valid."""
        errs: list[str] = []
        ids = [c.id for c in self.components]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            errs.append(f"duplicate component ids: {dupes}")
        known = set(ids)
        for e in self.edges:
            for end in (e.source, e.target):
                if end not in known:
                    errs.append(f"edge references unknown component {end!r}")
            if e.stream is not None:
                errs.extend(e.stream.validate())
        owners: dict[str, str] = {}
        for c in self.components:
            for s in c.states:
                if s.id in owners:
                    errs.append(
                        f"state {s.id!r} owned by both {owners[s.id]!r} "
                        f"and {c.id!r}; states need exactly one owner")
                owners[s.id] = c.id
        return errs
