"""Rule-based candidate generation (Tessera Planner, step 2).

Rules read region semantics and *verified* state contracts; they never
guess.  An unverified state pins its writer nodes into one stage on one
device (conservative default), which is exactly why contract probing pays:
verification unlocks transformations.

v0 rules (exact-only):
  R1 single-device baseline           (always)
  R2 co-located multi-GPU group       (if any region admits parallel_degree>1)
  R3 stage disaggregation             (state-disjoint node groups -> devices)
  R4 disaggregation + higher degree   (R3 with parallel groups per stage)
"""

from __future__ import annotations

from ..graph.program import Program
from ..graph.regions import Node
from .plan import DeploymentPlan, Hardware, OverlapMode, Stage


def _state_groups(program: Program) -> list[list[Node]]:
    """Partition nodes into groups that must co-locate: nodes sharing any
    non-recomputable, session-scoped state belong together (union-find)."""
    nodes = list(program.root.iter_nodes())
    parent = {n.id: n.id for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    by_state: dict[str, list[Node]] = {}
    for node in nodes:
        for sid in [*node.reads, *node.writes]:
            spec = program.states.get(sid)
            if spec is None:
                continue
            binding = not spec.recomputable  # recomputable features don't bind
            # Unverified contracts are treated as binding regardless of flags.
            if not spec.verified:
                binding = True
            if binding:
                by_state.setdefault(sid, []).append(node)
    for group in by_state.values():
        for other in group[1:]:
            union(group[0].id, other.id)

    roots: dict[str, list[Node]] = {}
    for node in nodes:
        roots.setdefault(find(node.id), []).append(node)
    # Preserve program order within and across groups.
    order = {n.id: i for i, n in enumerate(nodes)}
    groups = sorted(roots.values(), key=lambda g: min(order[n.id] for n in g))
    for g in groups:
        g.sort(key=lambda n: order[n.id])
    return groups


def _max_degree(program: Program, hardware: Hardware) -> int:
    """Largest legal parallel degree hinted by region attrs (v0: chunk_size
    divisors), capped by GPU count."""
    degree = 1
    for region in program.root.iter_regions():
        chunk = region.attrs.get("chunk_size")
        if isinstance(chunk, int):
            for d in range(min(chunk, hardware.num_gpus), 0, -1):
                if chunk % d == 0:
                    degree = max(degree, d)
                    break
    return max(1, min(degree, hardware.num_gpus))


def generate_candidates(program: Program, hardware: Hardware) -> list[DeploymentPlan]:
    groups = _state_groups(program)
    all_ids = [n.id for g in groups for n in g]
    plans: list[DeploymentPlan] = []

    # R1: single-device reference-shaped baseline.
    plans.append(DeploymentPlan(
        id="baseline_1gpu",
        stages=[Stage(id="all", node_ids=all_ids, device=0)],
        transforms=["placement:single"],
        notes="reference placement; fallback anchor"))

    # R2: co-located group at the highest legal degree.
    degree = _max_degree(program, hardware)
    if degree > 1:
        plans.append(DeploymentPlan(
            id=f"colocated_deg{degree}",
            stages=[Stage(id="all", node_ids=all_ids, device=0,
                          parallel_degree=degree)],
            transforms=["placement:colocate", f"parallel:degree{degree}"],
            notes="latency-family: whole budget shortens the critical path"))

    # R3: disaggregate state-disjoint groups onto separate devices.
    if len(groups) > 1 and hardware.num_gpus >= len(groups):
        stages = [
            Stage(id=f"stage{i}", node_ids=[n.id for n in g], device=i,
                  overlap=OverlapMode.CROSS_CHUNK if i > 0 else OverlapMode.NONE)
            for i, g in enumerate(groups)
        ]
        plans.append(DeploymentPlan(
            id=f"disagg_{len(groups)}stage",
            stages=stages,
            transforms=["placement:disaggregate", "overlap:cross_chunk"],
            notes="throughput-family: period drops from sum to max"))

        # R4: disaggregation with parallel first stage if budget remains.
        spare = hardware.num_gpus - len(groups)
        if spare >= 1 and degree > 1:
            first_deg = min(degree, spare + 1)
            stages4 = [Stage(id="stage0", node_ids=[n.id for n in groups[0]],
                             device=0, parallel_degree=first_deg)]
            for i, g in enumerate(groups[1:], start=1):
                stages4.append(Stage(
                    id=f"stage{i}", node_ids=[n.id for n in g],
                    device=first_deg + i - 1, overlap=OverlapMode.CROSS_CHUNK))
            plans.append(DeploymentPlan(
                id=f"disagg_par{first_deg}",
                stages=stages4,
                transforms=["placement:disaggregate", "overlap:cross_chunk",
                            f"parallel:degree{first_deg}"],
                notes="combined family: parallel bottleneck stage + pipeline"))
    return plans
