"""Sequential executor for IR graphs.

The ``SequentialExecutor`` runs an ``IRGraph``'s operators in topological order
on a single device. It is the reference execution strategy: no parallelism, no
scheduling optimization. Its purpose is correctness validation -- the agent
constructs an IR from the user's pipeline, then runs this executor and
compares outputs against the reference to verify the IR is faithful.

Operator implementations live on the operator itself (``op.execute``), so the
executor takes only the graph. Persistent state is kept in a dict inside the
executor and handed to each operator through a ``_ScopedStateStore`` that
enforces the operator's declared ``state_reads`` / ``state_writes``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from wllm.serving.ir.graph import IRGraph, IROperator, StateStore


class _ScopedStateStore:
    """StateStore implementation scoped to one operator's declared state.

    ``get(name)`` requires ``name`` in ``op.state_reads | op.state_writes``.
    ``set(name, value)`` requires ``name`` in ``op.state_writes``.

    In-place mutation happens on the object returned by ``get``; there is no
    attempt to enforce that a read-only caller refrains from mutating the
    returned object (too expensive for tensor state, and the declaration is
    what the analysis tools rely on regardless).
    """

    def __init__(self, op: IROperator, state: dict[str, Any]) -> None:
        self._op = op
        self._state = state
        self._allowed_reads = set(op.state_reads) | set(op.state_writes)
        self._allowed_writes = set(op.state_writes)

    def get(self, name: str) -> Any:
        if name not in self._allowed_reads:
            raise RuntimeError(
                f"Operator {self._op.name!r} attempted to get state {name!r}, "
                f"but {name!r} is not declared in its state_reads or state_writes. "
                f"Declared reads: {sorted(self._allowed_reads - self._allowed_writes)}, "
                f"declared writes: {sorted(self._allowed_writes)}."
            )
        if name not in self._state:
            raise RuntimeError(
                f"Operator {self._op.name!r} attempted to get state {name!r}, "
                f"but no value has been stored for it yet. Either pre-populate "
                f"via SequentialExecutor.init_state({{...}}) or ensure an "
                f"earlier operator calls state.set({name!r}, ...)."
            )
        return self._state[name]

    def set(self, name: str, value: Any) -> None:
        if name not in self._allowed_writes:
            raise RuntimeError(
                f"Operator {self._op.name!r} attempted to set state {name!r}, "
                f"but {name!r} is not declared in its state_writes. "
                f"Declared writes: {sorted(self._allowed_writes)}."
            )
        self._state[name] = value


class SequentialExecutor:
    """Executes an IRGraph sequentially in topological order.

    Usage:
        graph = build_some_graph(cfg)
        executor = SequentialExecutor(graph)

        # Optional: pre-populate persistent state (e.g., empty KV caches
        # allocated at session start).
        executor.init_state({"step_0_kv": empty_cache(), ...})

        # Session init
        executor.run_preamble({"init_data": tensor}, context)

        # Per-chunk execution
        for chunk in chunks:
            outputs = executor.run_chunk({"audio": chunk}, context)

    Each operator is a subclass of ``IROperator`` (or produced by the
    ``@ir_operator`` decorator) with an ``execute(inputs, context, state)``
    method. The executor routes tensor inputs from upstream ports, hands the
    operator a ``StateStore`` scoped to its declared state dependencies,
    invokes ``execute``, and routes the returned outputs to downstream
    operators.

    The ``context`` is an opaque object carrying whatever scalar / config
    state the operators need (session state, model runner handles, per-chunk
    parameters, etc.). The executor passes it through without interpretation.
    Persistent tensor state (KV caches, cached latents) should live in the
    executor's state dict, accessed via the StateStore.
    """

    def __init__(self, graph: IRGraph) -> None:
        self.graph = graph

        errors = graph.validate()
        if errors:
            joined = "\n  - ".join(errors)
            raise ValueError(
                f"Graph {graph.name!r} failed validation:\n  - {joined}"
            )

        self._preamble_order = self._compute_topo_order(graph.preamble_ops)
        self._chunk_order = self._compute_topo_order(graph.chunk_ops)

        self._input_map = self._build_input_map()

        self._state: dict[str, Any] = {}

    def init_state(self, state: dict[str, Any]) -> None:
        """Pre-populate persistent state before any operators run.

        Use this for state allocated externally (e.g., empty KV caches
        created at session start). Preamble operators can also populate
        state by calling ``state.set(...)`` during ``run_preamble``.

        Every name passed here must correspond to a declared ``StateObject``
        in the graph.
        """
        for name in state:
            if name not in self.graph.state_objects:
                raise ValueError(
                    f"init_state: {name!r} is not a declared StateObject in "
                    f"graph {self.graph.name!r}. Declared: "
                    f"{sorted(self.graph.state_objects)}."
                )
        self._state.update(state)

    @property
    def state(self) -> dict[str, Any]:
        """Read-only view of the current persistent state dict.

        Callers can inspect state values (e.g., to assert KV cache sizes
        after a run), but should go through ``init_state`` to mutate.
        """
        return dict(self._state)

    def _compute_topo_order(self, op_names: list[str]) -> list[str]:
        """Kahn's algorithm over the subgraph induced by op_names."""
        op_set = set(op_names)
        if not op_set:
            return []

        adj: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {name: 0 for name in op_set}

        for edge in self.graph.edges:
            if edge.src_op in op_set and edge.dst_op in op_set:
                adj[edge.src_op].append(edge.dst_op)
                in_degree[edge.dst_op] += 1

        queue = [n for n in op_names if in_degree[n] == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for succ in adj[node]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(order) != len(op_set):
            remaining = op_set - set(order)
            raise ValueError(
                f"Cycle detected in operator dependencies. "
                f"Unresolvable operators: {sorted(remaining)}"
            )

        return order

    def _build_input_map(self) -> dict[str, dict[str, tuple[str, str]]]:
        """For each (op, input_port), find the (source_op, source_port) that feeds it.

        Only data edges (with both ports set) contribute to the input map.
        Ordering edges are used for topological sort but don't route data.

        Returns: {dst_op: {dst_port: (src_op, src_port)}}
        """
        mapping: dict[str, dict[str, tuple[str, str]]] = defaultdict(dict)
        for edge in self.graph.edges:
            if edge.is_data_edge:
                mapping[edge.dst_op][edge.dst_port] = (edge.src_op, edge.src_port)
        return dict(mapping)

    def _gather_inputs(
        self,
        op_name: str,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve input port values for an operator from the values dict.

        For each input port of the operator:
        1. Check if there's an edge feeding it -> use the source op's output
        2. Check if there's an external input with matching name -> use that
        3. Raise an error if the input cannot be resolved
        """
        op = self.graph.operators[op_name]
        inputs = {}
        edge_map = self._input_map.get(op_name, {})

        for port in op.inputs:
            if port.name in edge_map:
                src_op, src_port = edge_map[port.name]
                key = f"{src_op}.{src_port}"
                if key not in values:
                    raise RuntimeError(
                        f"Input {port.name!r} of operator {op_name!r} expects "
                        f"output {src_port!r} from {src_op!r}, but it is not "
                        f"available. Check topological ordering."
                    )
                inputs[port.name] = values[key]
            elif port.name in values:
                inputs[port.name] = values[port.name]
            else:
                raise RuntimeError(
                    f"Cannot resolve input {port.name!r} for operator {op_name!r}. "
                    f"No edge feeds it and no external input matches."
                )

        return inputs

    def _run_ops(
        self,
        op_order: list[str],
        external_inputs: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        """Execute a sequence of operators, returning all produced values."""
        values = dict(external_inputs)

        for op_name in op_order:
            op = self.graph.operators[op_name]

            if not op.should_run(context):
                continue

            inputs = self._gather_inputs(op_name, values)
            store = _ScopedStateStore(op, self._state)
            outputs = op.execute(inputs, context, store)

            if not isinstance(outputs, dict):
                raise RuntimeError(
                    f"Operator {op_name!r}.execute() must return a dict, "
                    f"got {type(outputs).__name__}."
                )

            for port in op.outputs:
                if port.name not in outputs:
                    raise RuntimeError(
                        f"Operator {op_name!r} did not produce expected output "
                        f"{port.name!r}. Got keys: {sorted(outputs.keys())}"
                    )
                values[f"{op_name}.{port.name}"] = outputs[port.name]

        return values

    def run_preamble(
        self,
        external_inputs: dict[str, Any] | None = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Execute session-initialization operators.

        These run once per session (not per chunk). Examples: text encoder
        prefill, image encoding, KV cache initialization.

        Returns all values produced during preamble execution.
        """
        return self._run_ops(
            self._preamble_order,
            external_inputs or {},
            context,
        )

    def run_chunk(
        self,
        external_inputs: dict[str, Any],
        context: Any = None,
    ) -> dict[str, Any]:
        """Execute one period of the chunk-periodic graph.

        This is the main execution method, called once per chunk. It runs
        chunk_ops in topological order, resolving inputs from edges and
        external_inputs, and returns the external outputs.

        Args:
            external_inputs: Tensor values for the graph's external_inputs
                (e.g., {"audio_chunk": tensor, "latent_seed": tensor}).
            context: Opaque object passed to each operator implementation.
                The executor does not interpret it.

        Returns:
            Dict mapping external output port names to their tensor values.
        """
        values = self._run_ops(
            self._chunk_order,
            external_inputs,
            context,
        )

        result = {}
        for port in self.graph.external_outputs:
            found = False
            for op_name in reversed(self._chunk_order):
                key = f"{op_name}.{port.name}"
                if key in values:
                    result[port.name] = values[key]
                    found = True
                    break
            if not found and port.name in values:
                result[port.name] = values[port.name]
                found = True
            if not found:
                raise RuntimeError(
                    f"External output {port.name!r} was not produced by any operator"
                )

        return result

    @property
    def chunk_execution_order(self) -> list[str]:
        """The topological order in which chunk operators execute."""
        return list(self._chunk_order)

    @property
    def preamble_execution_order(self) -> list[str]:
        """The topological order in which preamble operators execute."""
        return list(self._preamble_order)
