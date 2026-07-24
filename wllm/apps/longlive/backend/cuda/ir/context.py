"""Execution context for the LongLive IR operators.

Carries the (single-GPU) generation core whose runners/timesteps/noise
generator the operators drive. The executor passes this object through
unchanged to every ``op.execute``.
"""
from __future__ import annotations

import torch

from wllm.serving.rt_config import RTConfig
from wllm.apps.longlive.backend.cuda.pipeline import LongLiveCore


class LongLiveIRContext:
    def __init__(self, cfg: RTConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.core = LongLiveCore(cfg, device)

    # Convenience pass-throughs used by ops via ctx.core.
    def init_state_dict(self):
        """Initial persistent-state dict for SequentialExecutor.init_state.

        The heavy caches (DiT KV ring, VAE feat cache) live on the runners;
        the state entries below are the IR-visible handles/markers the
        operators declare and mutate in place.
        """
        from wllm.apps.longlive.backend.cuda import generation as G
        return {
            "kv_ring": self.core.dit_runner.kv_memory,
            "ring_state": G.new_ring_state(),
            "encoder_kv": None,
            "vae_cache": {"count": 0},
            "video_out": [],
        }
