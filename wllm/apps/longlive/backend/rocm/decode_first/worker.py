"""Variant `decode_first`: single-GPU scheduling micro-opt.

The IR shows `cache_write ∥ vae_decode_*` are independent (cache-write persists
this chunk's clean K/V for the *next* chunk; it is off the current chunk's
first-frame critical path). The reference does cache_write *before* any VAE
decode, so the first user-visible frame waits an extra DiT pass. This variant
decodes frame 0 *before* the cache-write, cutting ~1 DiT pass off the
first-frame latency, then does the cache-write, then decodes the rest.

Same total work and same output (the reorder touches disjoint state:
`vae_feat_cache` vs `kv_cache`), so it is bit-faithful to the reference; it only
moves the first frame earlier. Isolates the decode-before-cache lever.
"""

from __future__ import annotations

import torch

from wllm.apps.longlive.backend.rocm.baseline_single.worker import Worker as BaselineWorker


class Worker(BaselineWorker):
    @torch.inference_mode()
    def _generate_chunk(self, write: bool = True):
        core = self.core
        idx = core.compute_chunk_indices()
        latents = core.sample_noise()
        for i in range(self._num_steps):
            latents = core.denoise_step(latents, i, idx)
        # decode frame 0 first (first frame out one DiT-pass earlier) ...
        frame0 = core.decode_frame(latents, 0)
        if write:
            self.video_buffer.write(frame0.cpu().numpy())
        # ... then persist this chunk's clean K/V for the next chunk ...
        core.cache_write(latents, idx)
        # ... then stream the remaining frames.
        for l in range(1, self._chunk_size):
            frame = core.decode_frame(latents, l)
            if write:
                self.video_buffer.write(frame.cpu().numpy())
        core.advance_chunk(idx)
