"""Variant: `baseline_single` — single-GPU, reference-faithful.

Reproduces the reference computation exactly (via ``LongLiveCore``) behind the
shared worker frontend, on one GPU. This is the agent-authored anchor that
proves ``LongLiveCore`` matches the reference; every multi-GPU variant then
parallelizes this same core, so its correctness pins down theirs.

No optimization here beyond the reference — it exists so the correctness
harness can compare "my code, 1 GPU" against "reference, 1 GPU" and attribute
any later delta purely to a parallelization lever.
"""

from __future__ import annotations

import torch

from wllm.serving.logger import init_logger
from wllm.serving.utils.rand import set_global_seed

from wllm.apps.longlive.backend.rocm.ir.engine import LongLiveCore
from wllm.apps.longlive.backend.rocm.worker_base import LongLiveWorkerBase

logger = init_logger(__name__)


class Worker(LongLiveWorkerBase):
    def _init_gen(self):
        self.core = LongLiveCore(self.cfg, self.device)
        self.video_buffer = self._create_video_buffer()
        self._num_steps = int(self.cfg.num_inference_steps)
        self._chunk_size = int(self.cfg.chunk_size)

    @torch.inference_mode()
    def _warmup(self):
        # Mirror reference start_instance + worker.warmup: warm the VAE, then
        # generate a full ring's worth of chunks so all shapes are compiled/
        # cudnn-benchmarked before the first user-visible chunk.
        set_global_seed(self.cfg.seed)
        self.core.reset()
        self.core.seed()
        self.core.set_prompt(self.cfg.prompt or "warmup")
        ring_capacity = max(self._chunk_size,
                            int(self.cfg.context_window_size) + self._chunk_size)
        warmup_blocks = (ring_capacity // self._chunk_size) + 1
        for _ in range(warmup_blocks):
            self._generate_chunk(write=False)
        self.core.reset()
        self.video_buffer.clear()

    def _reset_gen(self):
        self.core.reset()
        self.video_buffer.clear()

    @torch.inference_mode()
    def _apply_prompt(self, text: str, is_first: bool):
        if is_first:
            self.core.reset()
            self.core.seed()
            self.core.set_prompt(text)
            logger.info("LongLivePipeline.init_session: prompt=%r block_idx=%d",
                        text, self.core.block_idx)
        else:
            self.core.set_prompt(text)
            self.core.apply_scene_cut(text)
            logger.info("LongLivePipeline.update_prompt: block_idx=%d, prompt=%r",
                        self.core.block_idx, text)

    @torch.inference_mode()
    def _generate_chunk(self, write: bool = True):
        core = self.core
        idx = core.compute_chunk_indices()
        latents = core.sample_noise()
        for i in range(self._num_steps):
            latents = core.denoise_step(latents, i, idx)
        core.cache_write(latents, idx)
        for l in range(self._chunk_size):
            frame = core.decode_frame(latents, l)
            if write:
                self.video_buffer.write(frame.cpu().numpy())
        core.advance_chunk(idx)

    def _step(self):
        self._generate_chunk(write=True)

    def _teardown(self):
        if getattr(self, "video_buffer", None) is not None:
            self.video_buffer.unlink()
