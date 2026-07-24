"""Program: the top-level wGraph artifact.

A Program binds a region tree, its state contracts, its stream contracts,
and one quality contract into a single validatable unit.  `validate()`
returns hard errors; `warnings()` returns soft issues (e.g. unverified
state contracts) that the planner treats as restrictions, not failures:
an unverified state pins its nodes to conservative scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .quality import QualityContract
from .regions import Region
from .states import StateSpec
from .streams import StreamSpec


@dataclass
class Program:
    name: str
    root: Region
    states: dict[str, StateSpec] = field(default_factory=dict)
    streams: dict[str, StreamSpec] = field(default_factory=dict)
    quality: QualityContract = field(default_factory=QualityContract)
    meta: dict = field(default_factory=dict)   # model_id, source, commit, hw hints

    # ---------------------------------------------------------------- helpers
    def region_ids(self) -> set[str]:
        return {r.id for r in self.root.iter_regions()}

    def node_ids(self) -> set[str]:
        return {n.id for n in self.root.iter_nodes()}

    def endpoints(self) -> set[str]:
        """Ids a stream may attach to: any node or region."""
        return self.node_ids() | self.region_ids()

    # -------------------------------------------------------------- validation
    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.name:
            errs.append("program name must be non-empty")

        # Region/node tree, uniqueness.
        errs.extend(self.root.validate())
        seen_regions: set[str] = set()
        for region in self.root.iter_regions():
            if region.id in seen_regions:
                errs.append(f"duplicate region id '{region.id}'")
            seen_regions.add(region.id)
        seen_nodes: set[str] = set()
        for node in self.root.iter_nodes():
            if node.id in seen_nodes:
                errs.append(f"duplicate node id '{node.id}'")
            seen_nodes.add(node.id)
        overlap = seen_regions & seen_nodes
        for oid in sorted(overlap):
            errs.append(f"id '{oid}' used by both a region and a node")

        # States: spec validity, key consistency, references, owner existence.
        for key, spec in self.states.items():
            if key != spec.id:
                errs.append(f"state map key '{key}' != spec id '{spec.id}'")
            errs.extend(spec.validate())
            if spec.owner and spec.owner not in self.endpoints():
                errs.append(f"state '{spec.id}': owner '{spec.owner}' not found")
        declared = set(self.states)
        for node in self.root.iter_nodes():
            for sid in [*node.reads, *node.writes]:
                if sid not in declared:
                    errs.append(f"node '{node.id}' references undeclared state '{sid}'")
        # Exactly one writer per ordered state (unless owner marks delegation).
        writers: dict[str, list[str]] = {}
        for node in self.root.iter_nodes():
            for sid in node.writes:
                writers.setdefault(sid, []).append(node.id)
        for sid, wlist in writers.items():
            spec = self.states.get(sid)
            if spec is not None and spec.ordered and len(wlist) > 1:
                errs.append(
                    f"ordered state '{sid}' has multiple writers {sorted(wlist)}; "
                    "declare a single owner or mark the state unordered"
                )

        # Streams: spec validity, key consistency, endpoint existence.
        eps = self.endpoints()
        for key, spec in self.streams.items():
            if key != spec.id:
                errs.append(f"stream map key '{key}' != spec id '{spec.id}'")
            errs.extend(spec.validate())
            for side, ep in (("producer", spec.producer), ("consumer", spec.consumer)):
                if ep and ep not in eps:
                    errs.append(f"stream '{spec.id}': {side} '{ep}' not found")

        errs.extend(self.quality.validate())
        return errs

    def warnings(self) -> list[str]:
        warns: list[str] = []
        for spec in self.states.values():
            if not spec.verified:
                warns.append(
                    f"state '{spec.id}' contract is unverified — planner will "
                    "treat it as ordered/non-migratable/non-forkable"
                )
        for spec in self.streams.values():
            if spec.deadline_ms is None and spec.rate_hz is not None:
                warns.append(
                    f"stream '{spec.id}' has a nominal rate but no deadline; "
                    "smoothness will be reported but not enforced"
                )
        return warns

    # ---------------------------------------------------------------- summary
    def summary(self) -> str:
        lines = [f"Program '{self.name}'"]
        for region in self.root.iter_regions():
            depth = _depth_of(self.root, region.id)
            pad = "  " * depth
            lines.append(f"{pad}- region {region.id} [{region.kind.value}]"
                         f"{' attrs=' + str(region.attrs) if region.attrs else ''}")
            for node in region.nodes:
                io = ""
                if node.reads or node.writes:
                    io = f" (r:{','.join(node.reads)} w:{','.join(node.writes)})"
                lines.append(f"{pad}    * node {node.id} [{node.op.value}]{io}")
        lines.append(f"states: {len(self.states)} "
                     f"(verified {sum(s.verified for s in self.states.values())})")
        lines.append(f"streams: {len(self.streams)}")
        lines.append(f"quality: {self.quality.mode.value}")
        return "\n".join(lines)


def _depth_of(root: Region, region_id: str, depth: int = 0) -> int:
    if root.id == region_id:
        return depth
    for child in root.children:
        found = _depth_of(child, region_id, depth + 1)
        if found >= 0:
            return found
    return -1
