"""Deployment knobs for the orchestrated Krea+SAM backend.

A *variant* is one concrete ``BackendConfig`` (a process topology +
scheduling choice). The same engine code (``coordinator.py``,
``sam_service.py``, ``krea_service.py``) runs every variant; what differs
is this config. Every variant is still launched, correctness-checked and
benchmarked independently — the config is what makes each row in the
variant queue a distinct measured experiment.

Knobs map directly onto IR-analysis findings:
  * ``sam_gpu`` / ``krea_gpus`` — the worker-graph pipeline stages
    (krea_v2v ‖ sam_segment) placed on their own device(s).
  * ``krea_sp`` (= len(krea_gpus)) — sequence parallelism inside the DiT
    (a below-IR-granularity within-chunk model-parallel lever).
  * ``stream_decode`` — emit decoded frames per-frame (the streaming
    edge vae_decode→composite), attacking latency-to-first-output.
  * ``sam_compile`` — black-box SAM torch.compile knob.
  * ``krea_pipeline_gpus`` — split the Krea model-graph stages
    (encode|DiT|decode) across devices and pipeline across chunks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class BackendConfig:
    # device placement
    sam_gpu: int = 1
    krea_gpus: List[int] = field(default_factory=lambda: [2])

    # Krea internal model-graph pipeline. None => monolithic Krea on
    # krea_gpus. "vae_dit" => split into a VAE service (encode+decode) on
    # krea_gpus[0] and a DiT service (denoise) on krea_gpus[1], pipelined
    # across chunks.
    krea_pipeline: Optional[str] = None
    krea_pipeline_gpus: Optional[List] = None

    # scheduling knobs
    stream_decode: bool = False          # coordinator writes composited frames one-by-one
    krea_stream_frames: bool = False     # Krea service streams each latent's decoded frames as produced
    sam_compile: bool = False            # torch.compile the SAM model

    # co-location flag: when True SAM and Krea share one GPU (reference-like
    # placement) but still run as concurrent processes.
    colocate_sam_krea: bool = False

    variant_name: str = "unnamed"

    @property
    def krea_sp(self) -> int:
        return len(self.krea_gpus)

    def all_gpus(self) -> List[int]:
        gpus = set(self.krea_gpus) | {self.sam_gpu}
        if self.krea_pipeline_gpus:
            for g in self.krea_pipeline_gpus:
                if isinstance(g, list):
                    gpus.update(g)
                else:
                    gpus.add(g)
        return sorted(gpus)

    def num_gpus(self) -> int:
        return len(self.all_gpus())

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(s: str) -> "BackendConfig":
        return BackendConfig(**json.loads(s))
