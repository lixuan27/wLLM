"""Stage-config parsing for the in-tree omni engine.

The schema is the one the apps already ship: ``async_chunk`` plus a
``stage_args`` list whose entries carry ``stage_id`` / ``stage_type`` /
``runtime.devices`` / ``engine_args`` (including a dotted
``scheduler_cls``) / ``final_output`` / ``default_sampling_params``.

The ``__WLLM_OMNI_ENGINE__`` placeholder in ``scheduler_cls`` resolves
to *this* package, so unrendered configs work natively here while
rendered configs work on any bound engine. Validation is fail-closed:
unknown stage types and unresolvable scheduler classes are errors, not
warnings.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PLACEHOLDER = "__WLLM_OMNI_ENGINE__"
SELF_PACKAGE = "wllm.omni"
STAGE_TYPES = ("llm", "diffusion", "codec", "vocoder", "custom")


@dataclass
class StageSpec:
    stage_id: int
    stage_type: str
    devices: str = "0"
    engine_args: dict = field(default_factory=dict)
    final_output: bool = False
    final_output_type: str = "text"
    default_sampling_params: dict = field(default_factory=dict)

    @property
    def scheduler_path(self) -> str:
        return str(self.engine_args.get("scheduler_cls") or "")


@dataclass
class StageConfig:
    async_chunk: bool = False
    stages: list[StageSpec] = field(default_factory=list)

    def final_stage(self) -> StageSpec:
        finals = [s for s in self.stages if s.final_output]
        if len(finals) != 1:
            raise ValueError(
                f"stage config must mark exactly one final_output stage, "
                f"found {len(finals)}")
        return finals[0]


def load_stage_config(path: str | Path) -> StageConfig:
    doc = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(doc, dict) or "stage_args" not in doc:
        raise ValueError(f"{path}: stage config needs a 'stage_args' list")
    stages: list[StageSpec] = []
    for i, raw in enumerate(doc["stage_args"] or []):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: stage_args[{i}] is not a mapping")
        stype = str(raw.get("stage_type") or "")
        if stype not in STAGE_TYPES:
            raise ValueError(
                f"{path}: stage_args[{i}] has unknown stage_type {stype!r}; "
                f"expected one of {STAGE_TYPES}")
        runtime = raw.get("runtime") or {}
        stages.append(StageSpec(
            stage_id=int(raw.get("stage_id", i)),
            stage_type=stype,
            devices=str(runtime.get("devices", "0")),
            engine_args=dict(raw.get("engine_args") or {}),
            final_output=bool(raw.get("final_output", False)),
            final_output_type=str(raw.get("final_output_type", "text")),
            default_sampling_params=dict(
                raw.get("default_sampling_params") or {}),
        ))
    if not stages:
        raise ValueError(f"{path}: stage_args is empty")
    cfg = StageConfig(async_chunk=bool(doc.get("async_chunk", False)),
                      stages=stages)
    cfg.final_stage()          # fail closed on 0 or >1 final stages
    return cfg


def resolve_scheduler(dotted: str):
    """Import the scheduler class from a dotted path.

    Resolves the engine-package placeholder to this package. Raises
    ImportError/AttributeError on failure — never substitutes a default
    silently.
    """
    if not dotted:
        raise ValueError("scheduler_cls is empty; stage config must name "
                         "its scheduler explicitly")
    dotted = dotted.replace(PLACEHOLDER, SELF_PACKAGE)
    module_path, _, cls_name = dotted.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)
