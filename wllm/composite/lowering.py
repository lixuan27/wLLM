"""Plan lowering: connect a DeploymentPlan to the composite runtime.

The planner speaks in logical stages on integer device indices; the
walk executor speaks in component -> device-label placements
(``"gpu<N>"`` / ``"cpu"``).  Lowering is the translation between the
two, and it is fail-closed: every mismatch between plan and graph —
unknown node ids, components left unassigned or claimed by two stages,
violated placement pins, cpu-domain components in GPU stages, features
the runtime cannot execute yet (parallel groups) — is reported as an
explicit problem instead of being papered over.  A report with
problems must never be executed; callers go through :func:`require`,
which raises with the full problem list, before handing the placement
to :class:`~wllm.composite.executor.WalkExecutor`.

Overlap modes are recorded, not executed: ``CROSS_CHUNK`` stages are
listed in the report so callers know the plan intends pipelining, but
lowering makes no claim that the synchronous executor overlaps them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..planner.plan import DeploymentPlan, Hardware, OverlapMode
from .graph import ComponentGraph


@dataclass
class LoweringReport:
    """Outcome of lowering a plan onto a component graph.

    ``problems`` empty means the plan lowered cleanly and ``placement``
    is safe to hand to the executor.  A non-empty ``problems`` list
    makes the whole report unusable for execution — partial fields are
    kept only as evidence for diagnostics.
    """

    # component id -> device label ("gpu<N>" / "cpu")
    placement: dict[str, str] = field(default_factory=dict)
    # component id -> stage id that assigned it
    stage_of: dict[str, str] = field(default_factory=dict)
    # empty == lowered cleanly
    problems: list[str] = field(default_factory=list)
    # stage ids with OverlapMode.CROSS_CHUNK — informational only; the
    # synchronous executor does not actually overlap them
    overlapped: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def lower_plan(graph: ComponentGraph, plan: DeploymentPlan,
               hardware: Hardware | None = None) -> LoweringReport:
    """Lower ``plan`` onto ``graph``; never guess, only report.

    Checks performed (each violation appends one problem string):

    * every node id in every stage names a component of the graph;
    * every component is assigned by exactly one stage;
    * device labels honor each component's ``placement_domain``
      ("fixed:<dev>" must match, "cpu" cannot live in a GPU stage);
    * ``parallel_degree > 1`` is refused — the composite runtime has no
      parallel-group execution yet and lowering will not pretend;
    * with ``hardware`` given, every device index a stage touches must
      exist.
    """
    report = LoweringReport()
    known = {c.id for c in graph.components}

    for st in plan.stages:
        label = f"gpu{st.device}"

        if st.parallel_degree > 1:
            report.problems.append(
                f"stage {st.id}: parallel_degree {st.parallel_degree} "
                f"not yet lowerable; refusing to pretend a parallel "
                f"group exists")
        if hardware is not None:
            top = st.device + max(st.parallel_degree, 1) - 1
            if top >= hardware.num_gpus:
                report.problems.append(
                    f"stage {st.id}: needs device {top} but hardware "
                    f"has {hardware.num_gpus} GPUs")
        if st.overlap is OverlapMode.CROSS_CHUNK:
            report.overlapped.append(st.id)

        for nid in st.node_ids:
            if nid not in known:
                report.problems.append(
                    f"stage {st.id}: node {nid!r} is not a component "
                    f"of graph {graph.name!r}")
                continue
            if nid in report.stage_of:
                report.problems.append(
                    f"component {nid!r} assigned by both stage "
                    f"{report.stage_of[nid]!r} and stage {st.id!r}; "
                    f"exactly one stage may assign it")
                continue
            report.stage_of[nid] = st.id

            domain = graph.component(nid).placement_domain
            if domain.startswith("fixed:"):
                pinned = domain.split(":", 1)[1]
                if label != pinned:
                    report.problems.append(
                        f"component {nid!r} is pinned to {pinned!r} "
                        f"but stage {st.id} lowers to {label!r}")
                    continue
                report.placement[nid] = pinned
            elif domain == "cpu":
                # Stage.device is an integer GPU index; the plan cannot
                # encode a cpu placement, so a cpu-domain component in
                # any stage is a mismatch, not something to lower away.
                report.problems.append(
                    f"component {nid!r} has placement_domain 'cpu' but "
                    f"stage {st.id} lowers to GPU label {label!r}; the "
                    f"plan cannot place cpu-domain components")
            elif domain in ("gpu", "any"):
                report.placement[nid] = label
            else:
                report.problems.append(
                    f"component {nid!r}: unknown placement_domain "
                    f"{domain!r}; refusing to guess its device")

    for nid in sorted(known - set(report.stage_of)):
        report.problems.append(
            f"component {nid!r} is not assigned by any stage of plan "
            f"{plan.id!r}")
    return report


def require(report: LoweringReport) -> dict[str, str]:
    """Return the placement of a clean report; raise listing problems.

    This is the single gate between lowering and execution: callers
    must not read ``report.placement`` around it when ``ok`` is False.
    """
    if not report.ok:
        raise ValueError(
            f"plan does not lower cleanly "
            f"({len(report.problems)} problems): "
            + "; ".join(report.problems))
    return report.placement
