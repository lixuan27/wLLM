"""Streaming + sequence-parallel Krea orchestrator (best-latency @ ≥3 GPU).

Combines the two winning latency levers:
  * SP=K Krea (DiT frame-SP + VAE-decoder width-tile) shortens the denoise, and
  * per-frame streaming emit (frame_stream) removes the whole-chunk buffering.

All ranks run denoise_chunk (SP) then decode each latent frame together (the VAE
decode is a world-group collective / width-tile); rank 0 reads each frame's mask,
composites and emits it incrementally. First output frame therefore needs only
the (SP-shortened) denoise + one decode + one SAM frame.
"""

from __future__ import annotations

import numpy as np

from wllm.apps.krea_sam.backend.rocm.engine.orchestrator_sp import KreaOrchestratorSP
from wllm.apps.krea_sam.backend.rocm.engine.orchestrator_stream import denoise_chunk


class KreaOrchestratorStreamSP(KreaOrchestratorSP):
    def _process_chunk(self, krea_input, raw, mask_start):
        denoised, block_idx = denoise_chunk(self.pipe, krea_input)  # all ranks (SP)
        if denoised is None:
            return
        drop = self._output_frame_skip_frames if self.is_lead else 0
        p = 0
        for frame_i in range(int(denoised.shape[2])):
            latent_i = denoised[:, :, frame_i:frame_i + 1, :, :].clone()
            is_first = (block_idx == 0 and frame_i == 0)
            decoded_i = self.pipe.vae_runner.run(latent_i, is_first)  # all ranks (decode collective/tile)
            if not self.is_lead:
                continue
            pix = np.asarray(decoded_i[0].cpu().numpy())              # [Tf, H, W, 3]
            for j in range(int(pix.shape[0])):
                if p < drop:
                    p += 1
                    continue
                orig = raw[p]
                mask = self.sam_link.read_masks(mask_start + p, 1)[0]
                m = (mask > 0).astype(np.uint8)[:, :, None]
                self.video_buffer.write(orig * m + pix[j] * (1 - m))
                p += 1
        if self.is_lead:
            self._output_frame_skip_frames = 0
