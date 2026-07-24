"""wLLM user-facing API (L1: wrap a Python callable in <30 lines).

    from wllm.api import Application

    app = Application.from_callable(run, example_inputs={"prompt": "..."})
    report = app.baseline(repeats=5)
    plans = app.optimize(objective="first-output-latency", num_gpus=4)

L1 wraps the callable as a single opaque node inside a wGraph Program;
deeper capture (L2/L3) replaces that node with a structured region tree
produced by the contract-lifting pipeline, at which point `optimize()`
starts discovering co-location/disaggregation/parallel families.  At L1
the honest optimization surface is placement/replication only — wLLM
never claims transformations it cannot see the legality of.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .graph.program import Program
from .graph.quality import QualityContract
from .graph.regions import Node, NodeOp, Region, RegionKind
from .graph.streams import Modality
from .planner.constraints import filter_plans
from .planner.plan import DeploymentPlan, Hardware
from .planner.rules import generate_candidates
from .profiling.report import BaselineReport, profile_thunk

_SUFFIX_MODALITY = {
    ".mp4": Modality.FRAME, ".mov": Modality.FRAME, ".webm": Modality.FRAME,
    ".png": Modality.FRAME, ".jpg": Modality.FRAME, ".jpeg": Modality.FRAME,
    ".wav": Modality.AUDIO, ".flac": Modality.AUDIO, ".mp3": Modality.AUDIO,
    ".npy": Modality.LATENT, ".npz": Modality.LATENT,
    ".json": Modality.CONTROL,
}


def infer_modality(value: Any) -> Modality:
    if isinstance(value, str):
        suffix = Path(value).suffix.lower()
        if suffix in _SUFFIX_MODALITY:
            return _SUFFIX_MODALITY[suffix]
        return Modality.TOKEN
    if isinstance(value, (int, float, bool)):
        return Modality.CONTROL
    if isinstance(value, (list, tuple)) and value and isinstance(
            value[0], (int, float)):
        return Modality.ACTION
    return Modality.LATENT


@dataclass
class PlanSet:
    plans: list[DeploymentPlan]
    rejected: list[tuple[DeploymentPlan, str]]
    objective: str

    def best(self) -> DeploymentPlan | None:
        return self.plans[0] if self.plans else None

    def report(self) -> str:
        lines = [f"objective={self.objective}; "
                 f"{len(self.plans)} kept / {len(self.rejected)} rejected"]
        lines += [f"  keep {p.id}: {p.notes}" for p in self.plans]
        lines += [f"  drop {p.id}: {why}" for p, why in self.rejected]
        return "\n".join(lines)


@dataclass
class Application:
    name: str
    program: Program
    entrypoint: Callable[..., Any]
    example_inputs: dict[str, Any] = field(default_factory=dict)
    capture_level: int = 1

    # ------------------------------------------------------------- creation
    @classmethod
    def from_callable(cls, fn: Callable[..., Any], *,
                      example_inputs: dict[str, Any],
                      name: str | None = None,
                      quality: QualityContract | None = None) -> "Application":
        app_name = name or getattr(fn, "__name__", "app")
        node = Node(id=f"{app_name}_entry", op=NodeOp.OPAQUE_RUNNER,
                    impl_ref=f"callable:{app_name}",
                    attrs={"input_modalities": {
                        k: infer_modality(v).value
                        for k, v in example_inputs.items()}})
        root = Region(id=f"{app_name}_root", kind=RegionKind.SEQUENTIAL,
                      nodes=[node])
        program = Program(name=app_name, root=root,
                          quality=quality or QualityContract())
        errs = program.validate()
        if errs:
            raise ValueError("invalid program: " + "; ".join(errs))
        return cls(name=app_name, program=program, entrypoint=fn,
                   example_inputs=dict(example_inputs))

    # ------------------------------------------------------------ baselining
    def baseline(self, *, repeats: int = 5, warmup: int = 1,
                 save_dir: str | Path | None = None) -> BaselineReport:
        thunk = lambda: self.entrypoint(**self.example_inputs)  # noqa: E731
        report = profile_thunk(self.name, thunk, repeats=repeats,
                               warmup=warmup,
                               meta={"capture_level": self.capture_level})
        if save_dir is not None:
            report.save(save_dir, tag="baseline")
        return report

    # ------------------------------------------------------------- planning
    def optimize(self, *, objective: str = "first-output-latency",
                 num_gpus: int = 1, hbm_gb: float = 141.0,
                 node_mem_bytes: dict[str, int] | None = None) -> PlanSet:
        hardware = Hardware(num_gpus=num_gpus,
                            hbm_bytes_per_gpu=int(hbm_gb * 1e9))
        candidates = generate_candidates(self.program, hardware)
        kept, rejected = filter_plans(candidates, self.program, hardware,
                                      node_mem_bytes)
        costs = {n.id: (n.cost_hint_ms or 0.0)
                 for n in self.program.root.iter_nodes()}
        key = ((lambda p: p.estimate(costs)[0])
               if objective == "first-output-latency"
               else (lambda p: p.estimate(costs)[1]))
        kept.sort(key=key)
        return PlanSet(plans=kept, rejected=rejected, objective=objective)

    # -------------------------------------------------------------- fallback
    def reference_run(self, **overrides: Any) -> Any:
        """Always-available correct path: call the user's own entrypoint."""
        inputs = {**self.example_inputs, **overrides}
        return self.entrypoint(**inputs)
