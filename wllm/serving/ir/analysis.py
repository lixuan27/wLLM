"""Analysis tools for IR graphs.

These tools operate on the generic IRGraph structure to derive scheduling
and parallelism strategies. They are the primary interface through which
the agent discovers optimization opportunities from the IR.

The key analyses:
1. Cross-chunk dependency analysis: which operators share persistent state
   and therefore cannot be freely reordered across chunks.
2. Pipeline stage partitioning: how to assign operators to devices for
   pipeline parallelism, respecting state constraints.
3. Streaming overlap detection: which operator pairs can overlap execution
   due to streaming edges.
4. Critical path analysis: the longest dependency chain through one chunk.
5. Graph summary: human-readable description of all discovered opportunities.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from wllm.serving.ir.graph import IRGraph, IROperator, StateObject, StreamingPattern


@dataclass
class CrossChunkDep:
    """A cross-chunk dependency between two operators via a shared state object."""

    op_a: str
    op_b: str
    state_name: str
    description: str = ""


@dataclass
class CrossChunkDepInfo:
    """Result of cross-chunk dependency analysis."""

    dependencies: list[CrossChunkDep]
    independent_pairs: list[tuple[str, str]]
    state_groups: dict[str, set[str]]


def analyze_cross_chunk_dependencies(graph: IRGraph) -> CrossChunkDepInfo:
    """Analyze which chunk operators have cross-chunk state dependencies.

    Two operators A and B have a cross-chunk dependency if they both
    read/write the same chunk_persistent state object. This means A in
    chunk N constrains B in chunk N+1 (or vice versa).

    Returns:
        CrossChunkDepInfo with:
        - dependencies: list of (op_a, op_b, state) triples
        - independent_pairs: operator pairs with NO shared persistent state
        - state_groups: {state_name: {op_names that touch it}}
    """
    chunk_op_set = set(graph.chunk_ops)

    persistent_states = {
        name
        for name, obj in graph.state_objects.items()
        if obj.scope == "chunk_persistent"
    }

    state_groups: dict[str, set[str]] = defaultdict(set)
    for op_name in graph.chunk_ops:
        op = graph.operators[op_name]
        for state_name in set(op.state_reads) | set(op.state_writes):
            if state_name in persistent_states:
                state_groups[state_name].add(op_name)

    dependencies = []
    for state_name, ops in state_groups.items():
        ops_list = sorted(ops)
        for i, op_a in enumerate(ops_list):
            for op_b in ops_list[i:]:
                dependencies.append(CrossChunkDep(
                    op_a=op_a,
                    op_b=op_b,
                    state_name=state_name,
                    description=(
                        f"{op_a} and {op_b} share persistent state "
                        f"{state_name!r} -> cross-chunk dependency"
                    ),
                ))

    dep_pairs = set()
    for dep in dependencies:
        dep_pairs.add((dep.op_a, dep.op_b))
        dep_pairs.add((dep.op_b, dep.op_a))

    independent_pairs = []
    chunk_ops_list = list(graph.chunk_ops)
    for i, op_a in enumerate(chunk_ops_list):
        for op_b in chunk_ops_list[i + 1 :]:
            if (op_a, op_b) not in dep_pairs:
                independent_pairs.append((op_a, op_b))

    return CrossChunkDepInfo(
        dependencies=dependencies,
        independent_pairs=independent_pairs,
        state_groups=dict(state_groups),
    )


@dataclass
class PipelineStage:
    """A group of operators that share persistent state and must be co-located."""

    operators: list[str]
    state_objects: set[str]
    stage_idx: int = 0


def find_pipeline_stages(graph: IRGraph) -> list[PipelineStage]:
    """Partition chunk operators into pipeline stages for pipeline parallelism.

    Pipeline stages are formed by grouping operators that share persistent
    state objects (they MUST be co-located on the same device since they
    access the same state). Stages are then ordered by the within-chunk
    data dependency chain.

    The result directly tells the agent how many GPUs can be used for
    pipeline parallelism and which operators go on each GPU.

    Returns:
        List of PipelineStage, ordered by dependency (stage 0 runs first).
    """
    dep_info = analyze_cross_chunk_dependencies(graph)

    # Union-Find to group operators by shared persistent state
    parent: dict[str, str] = {op: op for op in graph.chunk_ops}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for dep in dep_info.dependencies:
        if dep.op_a in parent and dep.op_b in parent:
            union(dep.op_a, dep.op_b)

    groups: dict[str, list[str]] = defaultdict(list)
    for op_name in graph.chunk_ops:
        groups[find(op_name)].append(op_name)

    stage_state: dict[str, set[str]] = {}
    for root, ops in groups.items():
        states: set[str] = set()
        for op_name in ops:
            op = graph.operators[op_name]
            for s in set(op.state_reads) | set(op.state_writes):
                if s in graph.state_objects and graph.state_objects[s].scope == "chunk_persistent":
                    states.add(s)
        stage_state[root] = states

    # Order stages by data dependency: build a DAG over stage roots
    op_to_root = {op: find(op) for op in graph.chunk_ops}
    stage_edges: set[tuple[str, str]] = set()
    for edge in graph.edges:
        if edge.src_op in op_to_root and edge.dst_op in op_to_root:
            src_root = op_to_root[edge.src_op]
            dst_root = op_to_root[edge.dst_op]
            if src_root != dst_root:
                stage_edges.add((src_root, dst_root))

    # Topo sort the stage roots
    roots = list(groups.keys())
    in_degree = {r: 0 for r in roots}
    adj: dict[str, list[str]] = defaultdict(list)
    for src, dst in stage_edges:
        adj[src].append(dst)
        in_degree[dst] += 1

    queue = [r for r in roots if in_degree[r] == 0]
    ordered_roots: list[str] = []
    while queue:
        node = queue.pop(0)
        ordered_roots.append(node)
        for succ in adj[node]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(ordered_roots) != len(roots):
        remaining = set(roots) - set(ordered_roots)
        ordered_roots.extend(sorted(remaining))

    stages = []
    for idx, root in enumerate(ordered_roots):
        stages.append(PipelineStage(
            operators=groups[root],
            state_objects=stage_state[root],
            stage_idx=idx,
        ))

    return stages


@dataclass
class StreamingOverlap:
    """An opportunity to overlap two operators connected by a streaming edge."""

    upstream_op: str
    downstream_op: str
    pattern: StreamingPattern
    src_chunk_size: int | None = None
    dst_chunk_size: int | None = None
    rate_conversion_ratio: float | None = None
    description: str = ""


def find_streaming_overlaps(graph: IRGraph) -> list[StreamingOverlap]:
    """Find operator pairs connected by streaming edges.

    A streaming edge means the upstream operator produces output
    incrementally. The downstream operator can begin processing before
    the upstream finishes, enabling latency overlap.

    For FIXED_RATE edges, the accumulation ratio tells the agent how
    many upstream chunks must accumulate before one downstream invocation
    — enabling quantitative scheduling decisions.

    For VARIABLE_RATE edges, the agent knows overlap is legal but must
    determine the buffering strategy from the reference code —
    only qualitative scheduling is possible.
    """
    overlaps = []

    for edge in graph.edges:
        if not edge.is_streaming:
            continue

        src_op = graph.operators.get(edge.src_op)
        dst_op = graph.operators.get(edge.dst_op)
        if src_op is None or dst_op is None:
            continue

        info = edge.streaming
        desc_parts = [f"{edge.src_op} streams to {edge.dst_op}"]

        if info.pattern == StreamingPattern.FIXED_RATE:
            ratio = info.rate_conversion_ratio
            if ratio is not None and ratio != 1.0:
                desc_parts.append(
                    f"fixed-rate: {info.src_chunk_size} -> "
                    f"{info.dst_chunk_size} (ratio {ratio:.2f})"
                )
            elif ratio == 1.0:
                desc_parts.append("fixed-rate: 1:1")
        else:
            desc_parts.append("variable-rate (content-dependent boundaries)")

        if info.description:
            desc_parts.append(info.description)

        overlaps.append(StreamingOverlap(
            upstream_op=edge.src_op,
            downstream_op=edge.dst_op,
            pattern=info.pattern,
            src_chunk_size=info.src_chunk_size,
            dst_chunk_size=info.dst_chunk_size,
            rate_conversion_ratio=info.rate_conversion_ratio,
            description="; ".join(desc_parts),
        ))

    return overlaps


def compute_critical_path(
    graph: IRGraph,
    op_latencies: dict[str, float],
) -> tuple[list[str], float]:
    """Find the critical path through one chunk's computation graph.

    The critical path is the longest (by latency) dependency chain from
    any source operator to any sink operator. It determines the minimum
    single-chunk latency -- no scheduling strategy can make one chunk
    faster than the critical path.

    For pipeline parallelism, the bottleneck is the STAGE with the highest
    total latency, not the whole critical path. Use find_pipeline_stages()
    with this function to compute stage latencies.

    Args:
        graph: The IR graph to analyze.
        op_latencies: Empirical latency per operator {op_name: seconds}.
            Operators not in this dict are assigned latency 0.

    Returns:
        (path, total_latency) where path is the list of operator names
        on the critical path and total_latency is the sum of their latencies.
    """
    chunk_set = set(graph.chunk_ops)
    if not chunk_set:
        return [], 0.0

    adj: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        if edge.src_op in chunk_set and edge.dst_op in chunk_set:
            adj[edge.src_op].append(edge.dst_op)

    # Topo sort
    in_deg = {op: 0 for op in chunk_set}
    for src in adj:
        for dst in adj[src]:
            in_deg[dst] += 1

    queue = [op for op in graph.chunk_ops if in_deg[op] == 0]
    topo: list[str] = []
    while queue:
        node = queue.pop(0)
        topo.append(node)
        for succ in adj[node]:
            in_deg[succ] -= 1
            if in_deg[succ] == 0:
                queue.append(succ)

    # Longest path via DP
    dist: dict[str, float] = {op: 0.0 for op in chunk_set}
    pred: dict[str, str | None] = {op: None for op in chunk_set}

    for op_name in topo:
        lat = op_latencies.get(op_name, 0.0)
        for succ in adj[op_name]:
            candidate = dist[op_name] + lat
            if candidate > dist[succ]:
                dist[succ] = candidate
                pred[succ] = op_name

    # Find the sink with maximum distance + its own latency
    best_end = max(
        chunk_set,
        key=lambda op: dist[op] + op_latencies.get(op, 0.0),
    )
    total = dist[best_end] + op_latencies.get(best_end, 0.0)

    # Trace back the path
    path = [best_end]
    current = best_end
    while pred[current] is not None:
        current = pred[current]
        path.append(current)
    path.reverse()

    return path, total


def summarize_graph(
    graph: IRGraph,
    op_latencies: dict[str, float] | None = None,
) -> str:
    """Generate a human-readable summary of the graph's structure and opportunities.

    This is the primary entry point for the agent. The summary includes:
    1. Operator inventory with types and state dependencies
    2. Within-chunk dependency chain
    3. Cross-chunk dependencies and independent operator pairs
    4. Pipeline stage partitioning (how many GPUs can be used)
    5. Streaming overlap opportunities
    6. Critical path (if latencies provided)
    """
    lines: list[str] = []
    lines.append(f"=== IR Graph Summary: {graph.name} ===")
    lines.append("")

    # 1. Operators
    lines.append(f"## Operators ({len(graph.operators)} total)")
    lines.append(f"  Preamble: {len(graph.preamble_ops)}  |  Chunk: {len(graph.chunk_ops)}")
    lines.append("")

    for op_name in graph.preamble_ops + graph.chunk_ops:
        op = graph.operators[op_name]
        phase = "preamble" if op_name in graph.preamble_ops else "chunk"
        in_ports = ", ".join(p.name for p in op.inputs) or "(none)"
        out_ports = ", ".join(p.name for p in op.outputs) or "(none)"
        reads = ", ".join(op.state_reads) or "(none)"
        writes = ", ".join(op.state_writes) or "(none)"
        cond = "  [conditional]" if type(op).should_run is not IROperator.should_run else ""
        lines.append(f"  {op_name} ({op.op_type.value}, {op.stream_mode.value}, {phase}){cond}")
        lines.append(f"    inputs:  {in_ports}")
        lines.append(f"    outputs: {out_ports}")
        lines.append(f"    state reads:  {reads}")
        lines.append(f"    state writes: {writes}")
        if op.sub_graph:
            lines.append(f"    sub-graph: {op.sub_graph}")
        lines.append("")

    # 2. State objects
    lines.append(f"## State Objects ({len(graph.state_objects)} total)")
    for name, obj in graph.state_objects.items():
        lines.append(f"  {name} (scope={obj.scope}): {obj.description}")
    lines.append("")

    # 3. Edges
    data_edges = [e for e in graph.edges if e.is_data_edge]
    ordering_edges = [e for e in graph.edges if e.is_ordering_edge]
    streaming_edges = [e for e in graph.edges if e.is_streaming]
    lines.append(
        f"## Edges ({len(graph.edges)} total: {len(data_edges)} data, "
        f"{len(ordering_edges)} ordering, {len(streaming_edges)} streaming)"
    )
    for edge in data_edges:
        suffix = ""
        if edge.is_streaming:
            info = edge.streaming
            if info.pattern == StreamingPattern.FIXED_RATE:
                r = info.rate_conversion_ratio
                suffix = f"  [fixed-rate: {info.src_chunk_size}->{info.dst_chunk_size}"
                if r is not None:
                    suffix += f", ratio={r:.2f}"
                suffix += "]"
            else:
                suffix = "  [variable-rate]"
        lines.append(f"  {edge.src_op}.{edge.src_port} -> {edge.dst_op}.{edge.dst_port}{suffix}")
    for edge in ordering_edges:
        lines.append(f"  {edge.src_op} -> {edge.dst_op}  (ordering only)")
    lines.append("")

    # 4. Cross-chunk dependency analysis
    dep_info = analyze_cross_chunk_dependencies(graph)
    lines.append("## Cross-Chunk Dependencies")
    if dep_info.dependencies:
        for dep in dep_info.dependencies:
            lines.append(f"  {dep.op_a} <-({dep.state_name})-> {dep.op_b}")
    else:
        lines.append("  (none -- all chunk operators are independent across chunks)")
    lines.append("")

    lines.append("## Cross-Chunk Independent Pairs")
    if dep_info.independent_pairs:
        for a, b in dep_info.independent_pairs:
            lines.append(f"  {a} || {b}  (can overlap across chunks)")
    else:
        lines.append("  (none)")
    lines.append("")

    # 5. Pipeline stages
    stages = find_pipeline_stages(graph)
    lines.append(f"## Pipeline Stages ({len(stages)} stages = {len(stages)} GPUs for pipeline parallelism)")
    for stage in stages:
        state_str = ", ".join(sorted(stage.state_objects)) if stage.state_objects else "(no persistent state)"
        ops_str = ", ".join(stage.operators)
        lines.append(f"  Stage {stage.stage_idx}: [{ops_str}]")
        lines.append(f"    Persistent state: {state_str}")
    lines.append("")

    # 6. Streaming overlaps
    overlaps = find_streaming_overlaps(graph)
    lines.append(f"## Streaming Overlaps ({len(overlaps)} opportunities)")
    for ov in overlaps:
        lines.append(f"  {ov.description}")
    if not overlaps:
        lines.append("  (none)")
    lines.append("")

    # 7. Critical path
    if op_latencies:
        path, total = compute_critical_path(graph, op_latencies)
        lines.append(f"## Critical Path (total: {total:.3f}s)")
        for op_name in path:
            lat = op_latencies.get(op_name, 0.0)
            lines.append(f"  {op_name}: {lat:.3f}s")
        lines.append("")

        # Stage latencies for pipeline throughput
        lines.append("## Pipeline Stage Latencies")
        bottleneck_lat = 0.0
        bottleneck_stage = 0
        for stage in stages:
            stage_lat = sum(op_latencies.get(op, 0.0) for op in stage.operators)
            lines.append(f"  Stage {stage.stage_idx}: {stage_lat:.3f}s ({', '.join(stage.operators)})")
            if stage_lat > bottleneck_lat:
                bottleneck_lat = stage_lat
                bottleneck_stage = stage.stage_idx
        lines.append(f"  Bottleneck: Stage {bottleneck_stage} ({bottleneck_lat:.3f}s)")
        lines.append(f"  Pipeline throughput: 1 chunk per {bottleneck_lat:.3f}s (after bubble)")
        lines.append("")

    # 8. Validation info
    errors = graph.validate()
    if errors:
        lines.append("## VALIDATION ERRORS")
        for err in errors:
            lines.append(f"  - {err}")
    else:
        lines.append("## Validation: PASS")
    lines.append("")

    return "\n".join(lines)
