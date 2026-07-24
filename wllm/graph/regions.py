"""wGraph regions and nodes.

A Region carries *execution semantics* (what kind of loop repeats, over
what), not just topology.  This is the load-bearing difference from a flat
DAG: an autoregressive decode loop, a diffusion denoise loop, a chunked
world rollout, and a closed feedback loop all "repeat", but they admit
different legal transformations, so the IR must keep them distinguishable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator


class RegionKind(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    AUTOREGRESSIVE = "autoregressive"      # h_{t+1}, y_t = f(h_t, y_{<t}, c)
    DIFFUSION = "diffusion"                # z_{k-1} = Phi(z_k, c, k)
    FLOW = "flow"
    CHUNK_ROLLOUT = "chunk_rollout"        # s_{t+1} = F(s_t, a_t), chunked
    HIERARCHICAL_ROLLOUT = "hierarchical_rollout"
    MULTI_AGENT = "multi_agent"
    FEEDBACK = "feedback"                  # a_t = pi(o_t, h_t); h via real obs
    ENVIRONMENT = "environment"            # simulator / real-world step
    COMPOSITION = "composition"            # join/composite of parallel branches


class NodeOp(str, Enum):
    ENCODER = "encoder"
    PROJECTOR = "projector"
    TOKENIZER = "tokenizer"
    TRANSFORMER = "transformer"
    MOE = "moe"
    CODEC = "codec"              # VAE / RAE / audio codec / vocoder
    POLICY_HEAD = "policy_head"
    ACTION_DECODER = "action_decoder"
    ENVIRONMENT_STEP = "environment_step"
    RENDERER = "renderer"
    COMPOSITOR = "compositor"
    PROBE = "probe"
    OPAQUE_RUNNER = "opaque_runner"  # L0: whole external process/server
    CUSTOM = "custom"


@dataclass
class Node:
    """A leaf computation with declared state access and placement freedom."""

    id: str
    op: NodeOp
    impl_ref: str = ""            # dotted path / backend handle; "" for abstract
    reads: list[str] = field(default_factory=list)    # state ids
    writes: list[str] = field(default_factory=list)   # state ids
    placement_domain: str = "gpu"   # gpu | cpu | any | fixed:<device>
    cost_hint_ms: float | None = None
    attrs: dict = field(default_factory=dict)

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.id:
            errs.append("node id must be non-empty")
        dup = set(self.reads) & set(self.writes)
        # a node may read-modify-write the same state; that is fine — no error.
        _ = dup
        return errs


# Per-kind required attrs: kind -> (attr name, human reason)
_REQUIRED_ATTRS: dict[RegionKind, list[tuple[str, str]]] = {
    RegionKind.AUTOREGRESSIVE: [("termination", "eos|fixed_len|external")],
    RegionKind.DIFFUSION: [("num_steps", "denoise step count")],
    RegionKind.FLOW: [("num_steps", "integration step count")],
    RegionKind.CHUNK_ROLLOUT: [("chunk_size", "frames/latents per rollout chunk")],
    RegionKind.HIERARCHICAL_ROLLOUT: [("levels", "coarse-to-fine level names")],
    RegionKind.MULTI_AGENT: [("partition_key", "agent identity field")],
    RegionKind.FEEDBACK: [
        ("max_staleness_ms", "observation freshness bound"),
        ("deadline_ms", "action readiness deadline"),
    ],
}


@dataclass
class Region:
    """A nested execution scope. children run under this region's loop
    semantics; ordering among children/nodes is given by streams plus the
    region kind (SEQUENTIAL implies declaration order)."""

    id: str
    kind: RegionKind
    children: list["Region"] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)
    description: str = ""

    def iter_regions(self) -> Iterator["Region"]:
        yield self
        for child in self.children:
            yield from child.iter_regions()

    def iter_nodes(self) -> Iterator[Node]:
        for region in self.iter_regions():
            yield from region.nodes

    def find_node(self, node_id: str) -> Node | None:
        for node in self.iter_nodes():
            if node.id == node_id:
                return node
        return None

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.id:
            errs.append("region id must be non-empty")
        for attr, why in _REQUIRED_ATTRS.get(self.kind, []):
            if attr not in self.attrs:
                errs.append(
                    f"region '{self.id}' ({self.kind.value}): missing attr "
                    f"'{attr}' ({why})"
                )
        num_steps = self.attrs.get("num_steps")
        if num_steps is not None and (not isinstance(num_steps, int) or num_steps < 1):
            errs.append(f"region '{self.id}': num_steps must be int >= 1")
        chunk = self.attrs.get("chunk_size")
        if chunk is not None and (not isinstance(chunk, int) or chunk < 1):
            errs.append(f"region '{self.id}': chunk_size must be int >= 1")
        for node in self.nodes:
            errs.extend(node.validate())
        for child in self.children:
            errs.extend(child.validate())
        return errs
