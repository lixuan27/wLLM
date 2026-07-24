"""Deployment plans: the planner's output, the runtime's input.

A plan is configuration, not code: stages map nodes to devices with an
overlap mode.  Analytic cost model (calibrated later by measurement):

    latency  ≈ sum of per-chunk critical-path stage times + transfer costs
    period   ≈ max over devices of the per-chunk work resident on it

Co-locating stages shortens the critical path (no transfer, bigger group);
disaggregating lets stage s of chunk j overlap stage s+1 of chunk j-1, so
the sustainable period drops from a sum to a max.  These two formulas are
the exact-family cost model v0; measured residuals refine them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class OverlapMode(str, Enum):
    NONE = "none"              # stage runs inline with predecessor on device
    CROSS_CHUNK = "cross_chunk"  # pipelined against the next chunk


@dataclass
class Stage:
    id: str
    node_ids: list[str]
    device: int                     # logical device index within the plan
    overlap: OverlapMode = OverlapMode.NONE
    parallel_degree: int = 1        # devices in this stage's group


@dataclass
class Hardware:
    num_gpus: int
    hbm_bytes_per_gpu: int
    interconnect_gbps: float = 400.0   # effective per-link


@dataclass
class DeploymentPlan:
    id: str
    stages: list[Stage]
    transforms: list[str] = field(default_factory=list)  # applied transform ids
    exact: bool = True
    notes: str = ""

    def devices_used(self) -> set[int]:
        used: set[int] = set()
        for st in self.stages:
            used.update(range(st.device, st.device + st.parallel_degree))
        return used

    # ------------------------------------------------------------ cost model
    def estimate(self, node_cost_ms: dict[str, float],
                 transfer_ms: float = 0.0) -> tuple[float, float]:
        """Return (chunk_latency_ms, sustainable_period_ms)."""
        stage_time: dict[str, float] = {}
        for st in self.stages:
            total = sum(node_cost_ms.get(n, 0.0) for n in st.node_ids)
            # v0: parallel degree divides stage time (sublinear factors come
            # from calibration later — planner orders families, measurement
            # decides winners).
            stage_time[st.id] = total / max(st.parallel_degree, 1)

        # Latency: all stages lie on the chunk critical path, plus one
        # transfer per device boundary between consecutive stages.
        latency = 0.0
        prev_device: int | None = None
        for st in self.stages:
            latency += stage_time[st.id]
            if prev_device is not None and st.device != prev_device:
                latency += transfer_ms
            prev_device = st.device

        # Period: per-device resident work; overlapped stages on distinct
        # devices process different chunks concurrently.
        device_load: dict[int, float] = {}
        for st in self.stages:
            device_load[st.device] = device_load.get(st.device, 0.0) + stage_time[st.id]
        period = max(device_load.values()) if device_load else 0.0
        return latency, period

    def validate(self, hardware: Hardware) -> list[str]:
        errs: list[str] = []
        if not self.stages:
            errs.append(f"plan '{self.id}': no stages")
        used = self.devices_used()
        if used and max(used) >= hardware.num_gpus:
            errs.append(
                f"plan '{self.id}': needs device {max(used)} but hardware has "
                f"{hardware.num_gpus} GPUs")
        seen: set[str] = set()
        for st in self.stages:
            for n in st.node_ids:
                if n in seen:
                    errs.append(f"plan '{self.id}': node '{n}' in multiple stages")
                seen.add(n)
        return errs
