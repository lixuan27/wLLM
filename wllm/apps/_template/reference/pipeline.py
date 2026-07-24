"""Sequential reference pipeline for <app>.

String the app's model stages together here, strictly one after another, on
one GPU. Reuse the shared runtime where it fits: model runners under
``wllm/runner/`` (DiT, VAE, text encoder), model implementations under
``wllm/models/``, and external engines (external AR/omni engines) for stages
served as black boxes. Clarity beats speed — this pipeline is the
correctness oracle the optimization agent validates its variants against.

``wllm/apps/worldplay/reference/pipeline.py`` is a worked example built
on ``wllm.pipeline.base.BasePipeline``.
"""

from __future__ import annotations

import torch

from wllm.serving.rt_config import RTConfig


class AppPipeline:  # TODO rename to <App>Pipeline
    def __init__(self, cfg: RTConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        # TODO construct model runners / engine clients

    def start_instance(self) -> None:
        # TODO load weights, allocate caches, create the output buffer(s)
        raise NotImplementedError

    def init_session(self, **session_inputs) -> None:
        # TODO per-session state (prompt encoding, first-frame condition, ...)
        raise NotImplementedError

    def step(self, **chunk_inputs):
        # TODO one chunk of work: run each stage in order, return the output
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def terminate_instance(self) -> None:
        raise NotImplementedError
