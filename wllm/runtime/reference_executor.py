"""Reference executor: the always-correct, never-fast fallback.

Executes a Program sequentially on one device, honoring each region's loop
semantics.  It is (a) the correctness oracle every optimized plan is
compared against, and (b) the terminal entry of the runtime fallback chain
(optimized -> last-known-good -> reference).

Node implementations are supplied via an ImplRegistry mapping node id to a
callable ``fn(inputs: dict, state: dict, ctx: dict) -> dict``.  The
executor owns iteration; impls own math.  State access is enforced against
the node's declared reads/writes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..graph.program import Program
from ..graph.regions import Node, Region, RegionKind

ImplFn = Callable[[dict, "ScopedState", dict], dict]


class StateAccessError(RuntimeError):
    pass


class ScopedState:
    """State view restricted to one node's declared reads/writes."""

    def __init__(self, node: Node, store: dict[str, Any]):
        self._node = node
        self._store = store
        self._readable = set(node.reads) | set(node.writes)
        self._writable = set(node.writes)

    def get(self, name: str) -> Any:
        if name not in self._readable:
            raise StateAccessError(
                f"node '{self._node.id}' read undeclared state '{name}'")
        return self._store.get(name)

    def set(self, name: str, value: Any) -> None:
        if name not in self._writable:
            raise StateAccessError(
                f"node '{self._node.id}' wrote undeclared state '{name}'")
        self._store[name] = value


@dataclass
class ImplRegistry:
    impls: dict[str, ImplFn] = field(default_factory=dict)

    def register(self, node_id: str, fn: ImplFn) -> None:
        self.impls[node_id] = fn

    def resolve(self, node_id: str) -> ImplFn:
        if node_id not in self.impls:
            raise KeyError(f"no implementation registered for node '{node_id}'")
        return self.impls[node_id]


class ReferenceExecutor:
    def __init__(self, program: Program, registry: ImplRegistry):
        errs = program.validate()
        if errs:
            raise ValueError("invalid program: " + "; ".join(errs))
        self.program = program
        self.registry = registry
        self.state: dict[str, Any] = {}

    # ------------------------------------------------------------------ api
    def init_state(self, values: dict[str, Any]) -> None:
        for key in values:
            if key not in self.program.states:
                raise KeyError(f"init_state: undeclared state '{key}'")
        self.state.update(values)

    def run(self, inputs: dict[str, Any], num_chunks: int = 1) -> dict[str, Any]:
        """Run the whole program; CHUNK_ROLLOUT regions iterate num_chunks."""
        ctx = {"num_chunks": num_chunks, "chunk_index": 0, "step_index": 0}
        data = dict(inputs)
        self._run_region(self.program.root, data, ctx)
        return data

    # ------------------------------------------------------------- internals
    def _run_region(self, region: Region, data: dict, ctx: dict) -> None:
        kind = region.kind
        if kind == RegionKind.CHUNK_ROLLOUT:
            for j in range(ctx["num_chunks"]):
                ctx["chunk_index"] = j
                self._run_body_once(region, data, ctx)
        elif kind in (RegionKind.DIFFUSION, RegionKind.FLOW):
            for k in range(int(region.attrs["num_steps"])):
                ctx["step_index"] = k
                self._run_body_once(region, data, ctx)
        elif kind == RegionKind.AUTOREGRESSIVE:
            max_len = int(region.attrs.get("max_len", 1))
            for t in range(max_len):
                ctx["step_index"] = t
                self._run_body_once(region, data, ctx)
                if data.pop("__stop__", False):
                    break
        else:
            # SEQUENTIAL / PARALLEL / COMPOSITION / ... : reference semantics
            # is plain declaration order (parallelism is a deployment choice,
            # not a correctness requirement).
            self._run_body_once(region, data, ctx)

    def _run_body_once(self, region: Region, data: dict, ctx: dict) -> None:
        for node in region.nodes:
            self._run_node(node, data, ctx)
        for child in region.children:
            self._run_region(child, data, ctx)

    def _run_node(self, node: Node, data: dict, ctx: dict) -> None:
        fn = self.registry.resolve(node.id)
        out = fn(dict(data), ScopedState(node, self.state), dict(ctx))
        if out is None:
            return
        if not isinstance(out, dict):
            raise TypeError(f"impl '{node.id}' must return dict|None")
        data.update(out)
