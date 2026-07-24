"""Constraint filtering (Tessera Planner, step 3).

Rejects plans before any GPU time is spent: memory capacity, device
count, state migratability/placement, and quality-contract gating of
approximate transforms.  Returns (kept, rejected_with_reasons) so the
search log always shows why a family disappeared — silent pruning is how
planners lose users' trust.
"""

from __future__ import annotations

from ..graph.program import Program
from ..graph.quality import QualityMode
from .plan import DeploymentPlan, Hardware


def filter_plans(
    plans: list[DeploymentPlan],
    program: Program,
    hardware: Hardware,
    node_mem_bytes: dict[str, int] | None = None,
) -> tuple[list[DeploymentPlan], list[tuple[DeploymentPlan, str]]]:
    node_mem = node_mem_bytes or {}
    kept: list[DeploymentPlan] = []
    rejected: list[tuple[DeploymentPlan, str]] = []

    state_mem: dict[str, int] = {
        sid: spec.memory_bytes or 0 for sid, spec in program.states.items()
    }
    writers: dict[str, list[str]] = {}
    for node in program.root.iter_nodes():
        for sid in node.writes:
            writers.setdefault(sid, []).append(node.id)

    for plan in plans:
        reason = _reject_reason(plan, program, hardware, node_mem, state_mem,
                                writers)
        if reason is None:
            kept.append(plan)
        else:
            rejected.append((plan, reason))
    return kept, rejected


def _reject_reason(
    plan: DeploymentPlan,
    program: Program,
    hardware: Hardware,
    node_mem: dict[str, int],
    state_mem: dict[str, int],
    writers: dict[str, list[str]],
) -> str | None:
    errs = plan.validate(hardware)
    if errs:
        return "; ".join(errs)

    # Approximate transforms need a bounded-degradation contract.
    if not plan.exact and program.quality.mode != QualityMode.BOUNDED_DEGRADATION:
        return "approximate plan under exact quality contract"

    # Memory: nodes + states resident per device must fit HBM.
    node_stage_device: dict[str, int] = {}
    device_bytes: dict[int, int] = {}
    for st in plan.stages:
        for nid in st.node_ids:
            node_stage_device[nid] = st.device
            device_bytes[st.device] = (
                device_bytes.get(st.device, 0) + node_mem.get(nid, 0))
    for sid, mem in state_mem.items():
        owner_nodes = writers.get(sid, [])
        if owner_nodes:
            dev = node_stage_device.get(owner_nodes[0], 0)
            device_bytes[dev] = device_bytes.get(dev, 0) + mem
    for dev, used in device_bytes.items():
        if used > hardware.hbm_bytes_per_gpu:
            return (f"OOM on device {dev}: {used / 1e9:.1f} GB > "
                    f"{hardware.hbm_bytes_per_gpu / 1e9:.1f} GB")

    # Non-migratable ordered state must not have its readers and writer
    # split across devices (v0 conservative check).
    for sid, spec in program.states.items():
        if spec.migratable:
            continue
        touching = {
            node_stage_device.get(n.id)
            for n in program.root.iter_nodes()
            if sid in n.reads or sid in n.writes
        }
        touching.discard(None)
        if len(touching) > 1:
            return (f"state '{sid}' is non-migratable but touched from "
                    f"devices {sorted(touching)}")
    return None
