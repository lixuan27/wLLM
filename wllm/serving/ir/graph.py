"""Core IR data structures for representing chunk-periodic computation graphs.

A chunk-periodic computation graph is one where the same operator topology
repeats for each chunk of input (e.g., each latent chunk in a video generation
pipeline). State objects (like KV caches) persist across chunks and create
cross-chunk dependencies that constrain scheduling.

The IR makes data dependencies and state dependencies explicit so that
analysis tools can mechanically derive parallelism and scheduling strategies.

An ``IROperator`` is both a node in the computation graph and its
implementation. Subclass it (or use the ``@ir_operator`` decorator in
``wllm.ir.ops_base``) and override ``execute`` to attach behavior. The
``SequentialExecutor`` invokes ``op.execute(inputs, context, state)`` directly;
there is no separate registry of operator implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class OpType(Enum):
    """Classification of an operator's visibility to the optimization agent.

    BLACK_BOX: The agent cannot inspect or modify the operator's internals.
        It can only schedule it relative to other operators and choose device
        placement. Examples: an engine-served LLM, an omni-engine TTS, ASR model.

    EXPOSED: The agent can see the operator's implementation, data flow, and
        memory layout. It can restructure the operator's execution (e.g.,
        distribute across GPUs, pipeline with other exposed operators).
        Examples: DiT denoising step, VAE decode, Wav2Vec2 feature extraction.

    COMPOSITE: A wrapper that contains a sub-graph of other operators.
        The agent can either treat it as opaque (schedule as a unit) or
        descend into the sub-graph for finer-grained optimization.
    """

    BLACK_BOX = "black_box"
    EXPOSED = "exposed"
    COMPOSITE = "composite"


class StreamMode(Enum):
    """How an operator produces its output.

    BATCH: The operator's full output is materialized before any downstream
        operator can consume it.

    STREAMING: The operator produces output incrementally in chunks.
        Downstream operators connected by a streaming edge can start
        processing before the upstream operator finishes.
    """

    BATCH = "batch"
    STREAMING = "streaming"


@dataclass(frozen=True)
class TensorPort:
    """A named input or output slot of an operator.

    Ports are the connection points for IREdge instances. Shape and dtype
    hints are optional documentation for the agent -- they are not enforced
    by the executor.
    """

    name: str
    shape_hint: tuple[str | int, ...] | None = None
    dtype_hint: str | None = None


@dataclass
class StateObject:
    """A persistent state that operators read from or write to across chunks.

    State objects are the mechanism through which cross-chunk dependencies
    arise. If operator A writes to state S in chunk N, and operator B reads
    from S in chunk N+1, then B(N+1) depends on A(N).

    Scope controls the state's lifecycle:
        "chunk_persistent": Accumulates across chunks (e.g., KV caches).
            Creates cross-chunk dependencies.
        "session_init": Written once during session initialization, then
            read-only during chunk processing. Does NOT create cross-chunk
            dependencies during the periodic phase.
        "ephemeral": Created and consumed within a single chunk execution.
            Does NOT create cross-chunk dependencies.
    """

    name: str
    description: str = ""
    scope: str = "chunk_persistent"


class StateStore(Protocol):
    """Gated access to the executor's state objects.

    An operator receives a StateStore scoped to its own declared state
    dependencies and uses it to read and write persistent state (KV caches,
    prefilled embeddings, cached latents, etc.).

    Contract:
      * ``get(name)`` returns the stored object; the caller may mutate it
        in place iff ``name`` is in the operator's ``state_writes``.
      * ``set(name, value)`` rebinds the stored object for ``name``.
      * Both operations raise ``RuntimeError`` if ``name`` is not in the
        operator's declared ``state_reads`` / ``state_writes``.

    In-place mutation (e.g., a DiT runner writing into a KV cache object) is
    supported via ``get``: the caller takes the object and mutates it. The
    ``state_writes=[name]`` declaration documents the mutation for analysis;
    runtime enforcement only catches accesses to undeclared names.
    """

    def get(self, name: str) -> Any: ...

    def set(self, name: str, value: Any) -> None: ...


class IROperator:
    """A computation node in the IR graph with an attached implementation.

    Each operator:
      * Consumes tensors from its input ports (connected via ``IREdge``).
      * Produces tensors on its output ports.
      * May read from and/or write to named state objects, accessed through
        the ``StateStore`` handed to ``execute``.

    Subclass ``IROperator`` and override ``execute`` (and optionally
    ``should_run``) to provide behavior. The ``@ir_operator`` decorator in
    ``wllm.ir.ops_base`` offers a function-style shorthand for leaf ops.

    Constructor arguments are keyword-only so subclasses can forward their
    own parameters to ``super().__init__(...)`` cleanly.
    """

    def __init__(
        self,
        *,
        name: str,
        op_type: OpType,
        inputs: list[TensorPort] | None = None,
        outputs: list[TensorPort] | None = None,
        state_reads: list[str] | None = None,
        state_writes: list[str] | None = None,
        stream_mode: StreamMode = StreamMode.BATCH,
        sub_graph: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.op_type = op_type
        self.inputs: list[TensorPort] = list(inputs or [])
        self.outputs: list[TensorPort] = list(outputs or [])
        self.state_reads: list[str] = list(state_reads or [])
        self.state_writes: list[str] = list(state_writes or [])
        self.stream_mode = stream_mode
        self.sub_graph = sub_graph
        self.config: dict[str, Any] = dict(config or {})

    def execute(
        self,
        inputs: dict[str, Any],
        context: Any,
        state: StateStore,
    ) -> dict[str, Any]:
        """Compute this operator's outputs.

        Subclasses override this method. Default raises NotImplementedError.

        Args:
            inputs: {port_name: value} for each input port.
            context: Opaque object the executor passes through unchanged.
                Carries per-chunk scalars, model runner handles, etc.
            state: StateStore scoped to this operator's declared state
                dependencies. Use ``state.get(name)`` / ``state.set(name, val)``.

        Returns:
            {port_name: value} for each output port.
        """
        del inputs, context, state  # interface parameters; unused in base
        raise NotImplementedError(
            f"Operator {self.name!r} ({type(self).__name__}) has no execute() "
            f"implementation. Subclass IROperator and override execute(), or "
            f"use the @ir_operator decorator."
        )

    def should_run(self, context: Any) -> bool:
        """Return True if this operator should run for the current invocation.

        Default: always runs. Override for chunk-conditional operators
        (e.g., a denoising-step-0 prefill that only runs on the first chunk).
        """
        del context  # interface parameter; unused in base
        return True

    def has_state_dependency_with(self, other: "IROperator") -> bool:
        """Check if this operator shares any state (read or write) with another."""
        my_state = set(self.state_reads) | set(self.state_writes)
        other_state = set(other.state_reads) | set(other.state_writes)
        return bool(my_state & other_state)

    def writes_state_of(self, other: "IROperator") -> bool:
        """Check if this operator writes state that the other reads or writes."""
        return bool(set(self.state_writes) & (set(other.state_reads) | set(other.state_writes)))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"op_type={self.op_type.value!r}, "
            f"stream_mode={self.stream_mode.value!r}, "
            f"state_reads={self.state_reads}, "
            f"state_writes={self.state_writes})"
        )


class StreamingPattern(Enum):
    """How predictable the chunk boundaries are on a streaming edge.

    FIXED_RATE: Both the producer's output unit size and the consumer's
        input unit size are known constants (derivable from config or
        code). The accumulation ratio is deterministic and the agent
        can compute precise buffering latency.

    VARIABLE_RATE: The producer streams output incrementally, but chunk
        boundaries are content-dependent or otherwise not statically
        determinable. The agent knows overlap is legal but cannot
        compute a precise accumulation ratio.
    """

    FIXED_RATE = "fixed_rate"
    VARIABLE_RATE = "variable_rate"


@dataclass
class StreamingInfo:
    """Describes the streaming behavior of a data edge.

    Attached to an IREdge to indicate that the upstream operator
    produces output incrementally and the downstream operator can
    begin consuming before the upstream finishes.

    For FIXED_RATE edges, src_chunk_size and dst_chunk_size give the
    producer and consumer unit sizes, enabling the agent to compute
    how many producer chunks must accumulate before one consumer
    invocation (the rate_conversion_ratio).

    For VARIABLE_RATE edges, these fields are None. The agent knows
    overlap is possible but must determine buffering strategy from
    the reference code.

    The description field is freeform — the agent fills it with
    whatever it discovers about the streaming behavior.
    """

    pattern: StreamingPattern
    src_chunk_size: int | None = None
    dst_chunk_size: int | None = None
    description: str = ""

    @property
    def rate_conversion_ratio(self) -> float | None:
        """Ratio of destination chunk size to source chunk size.

        Only meaningful for FIXED_RATE edges. Returns None for
        VARIABLE_RATE or if chunk sizes are not set.
        """
        if self.pattern != StreamingPattern.FIXED_RATE:
            return None
        if self.src_chunk_size is None or self.dst_chunk_size is None:
            return None
        if self.src_chunk_size == 0:
            return None
        return self.dst_chunk_size / self.src_chunk_size


@dataclass
class IREdge:
    """A dependency between two operators.

    Two kinds:
    - **Data edge**: connects an output port of one operator to an input port
      of another. Both src_port and dst_port are set to port names.
    - **Ordering edge**: enforces execution order without data transfer.
      Both src_port and dst_port are None (or empty string).

    The executor uses all edges (data + ordering) for topological sort,
    but only routes tensors for data edges.

    Streaming metadata (the `streaming` field) is set on edges where the
    upstream operator produces output incrementally. It distinguishes
    fixed-rate streaming (deterministic accumulation ratio) from
    variable-rate streaming (content-dependent boundaries).
    """

    src_op: str
    src_port: str | None
    dst_op: str
    dst_port: str | None
    streaming: StreamingInfo | None = None

    @property
    def is_data_edge(self) -> bool:
        """True if this edge carries data (has both ports set)."""
        return bool(self.src_port) and bool(self.dst_port)

    @property
    def is_ordering_edge(self) -> bool:
        """True if this edge only enforces ordering (no data transfer)."""
        return not self.is_data_edge

    @property
    def is_streaming(self) -> bool:
        """True if this edge has streaming metadata."""
        return self.streaming is not None


@dataclass
class IRGraph:
    """A chunk-periodic computation graph.

    The graph has two phases:
    1. Preamble: operators that run once per session (e.g., encoder prefill,
       image encoding). Listed in `preamble_ops`.
    2. Chunk body: operators that run once per chunk, repeating with the
       same topology. Listed in `chunk_ops`.

    State objects persist across both phases and across chunks. The analysis
    tools use state dependencies to derive cross-chunk constraints and
    parallelism opportunities.
    """

    name: str
    operators: dict[str, IROperator] = field(default_factory=dict)
    edges: list[IREdge] = field(default_factory=list)
    state_objects: dict[str, StateObject] = field(default_factory=dict)
    external_inputs: list[TensorPort] = field(default_factory=list)
    external_outputs: list[TensorPort] = field(default_factory=list)
    preamble_ops: list[str] = field(default_factory=list)
    chunk_ops: list[str] = field(default_factory=list)
    sub_graphs: dict[str, IRGraph] = field(default_factory=dict)

    def add_operator(self, op: IROperator) -> None:
        if op.name in self.operators:
            raise ValueError(f"Operator {op.name!r} already exists in graph {self.name!r}")
        self.operators[op.name] = op

    def add_edge(self, edge: IREdge) -> None:
        if edge.src_op not in self.operators:
            raise ValueError(
                f"Edge source {edge.src_op!r} not found in graph {self.name!r}"
            )
        if edge.dst_op not in self.operators:
            raise ValueError(
                f"Edge destination {edge.dst_op!r} not found in graph {self.name!r}"
            )
        if edge.is_data_edge:
            src_op = self.operators[edge.src_op]
            dst_op = self.operators[edge.dst_op]
            if not any(p.name == edge.src_port for p in src_op.outputs):
                raise ValueError(
                    f"Source port {edge.src_port!r} not found on operator {edge.src_op!r}"
                )
            if not any(p.name == edge.dst_port for p in dst_op.inputs):
                raise ValueError(
                    f"Destination port {edge.dst_port!r} not found on operator {edge.dst_op!r}"
                )
        self.edges.append(edge)

    def add_state(self, state: StateObject) -> None:
        if state.name in self.state_objects:
            raise ValueError(
                f"State object {state.name!r} already exists in graph {self.name!r}"
            )
        self.state_objects[state.name] = state

    def get_incoming_edges(self, op_name: str) -> list[IREdge]:
        return [e for e in self.edges if e.dst_op == op_name]

    def get_outgoing_edges(self, op_name: str) -> list[IREdge]:
        return [e for e in self.edges if e.src_op == op_name]

    def get_predecessors(self, op_name: str) -> set[str]:
        return {e.src_op for e in self.edges if e.dst_op == op_name}

    def get_successors(self, op_name: str) -> set[str]:
        return {e.dst_op for e in self.edges if e.src_op == op_name}

    def validate(self) -> list[str]:
        """Check graph consistency. Returns list of error messages (empty if valid)."""
        errors = []

        for op_name in self.preamble_ops:
            if op_name not in self.operators:
                errors.append(f"Preamble op {op_name!r} not found in operators")

        for op_name in self.chunk_ops:
            if op_name not in self.operators:
                errors.append(f"Chunk op {op_name!r} not found in operators")

        all_registered = set(self.preamble_ops) | set(self.chunk_ops)
        for op_name in self.operators:
            if op_name not in all_registered:
                errors.append(
                    f"Operator {op_name!r} not listed in preamble_ops or chunk_ops"
                )

        for op in self.operators.values():
            for state_name in op.state_reads + op.state_writes:
                if state_name not in self.state_objects:
                    errors.append(
                        f"Operator {op.name!r} references unknown state {state_name!r}"
                    )

        for op in self.operators.values():
            if op.sub_graph is not None and op.sub_graph not in self.sub_graphs:
                errors.append(
                    f"Operator {op.name!r} references unknown sub-graph {op.sub_graph!r}"
                )

        for edge in self.edges:
            if edge.src_op not in self.operators:
                errors.append(f"Edge source {edge.src_op!r} not in operators")
            if edge.dst_op not in self.operators:
                errors.append(f"Edge dest {edge.dst_op!r} not in operators")

        chunk_set = set(self.chunk_ops)
        if self._has_cycle(chunk_set):
            errors.append("Cycle detected in chunk_ops dependency graph")

        return errors

    def _has_cycle(self, op_set: set[str]) -> bool:
        """Detect cycles in the subgraph induced by op_set using DFS."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {name: WHITE for name in op_set}

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for edge in self.get_outgoing_edges(node):
                succ = edge.dst_op
                if succ not in op_set:
                    continue
                if color[succ] == GRAY:
                    return True
                if color[succ] == WHITE and dfs(succ):
                    return True
            color[node] = BLACK
            return False

        return any(color[n] == WHITE and dfs(n) for n in op_set)
