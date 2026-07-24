"""Frame-streaming Krea orchestrator (latency lever).

The reference (and sam_parallel) emit a whole decoded chunk at once, so the
first output frame only appears after the ENTIRE chunk's SAM (12 frames) +
decode + composite finish. But the IR marks `vae_decode -> composite` as a
STREAMING 1:1 edge: each decoded pixel frame can be composited + emitted as
soon as its per-frame SAM mask is ready. Since SAM processes frames in order,
the first output frame then needs only the denoise + one decode + ONE SAM
frame, not the whole chunk — roughly halving latency-to-first-output.

Numerically identical output to sam_parallel (same frames, just emitted
incrementally); only the emission timing changes.
"""

from __future__ import annotations

import time

import numpy as np

from wllm.apps.krea_sam.backend.rocm.engine.orchestrator import KreaOrchestrator


def denoise_chunk(pipe, input_frames):
    """pipe.step minus the decode loop: returns (denoised_latents, block_idx)
    where block_idx is the value BEFORE the internal increment (needed for the
    per-frame is_first flag). Maintains pipe's cross-chunk state exactly."""
    block_idx = pipe._block_idx
    pixels = input_frames.permute(1, 0, 2, 3).unsqueeze(0).to(device=pipe.device, dtype=pipe.dtype)
    input_latents = pipe.vae_runner.encode(pixels, stream=block_idx > 0).to(
        device=pipe.device, dtype=pipe.dtype)
    chunk_frames = int(pipe.cfg.chunk_size)
    if int(input_latents.shape[2]) < chunk_frames:
        return None, block_idx
    input_latents = input_latents[:, :, -chunk_frames:].contiguous()
    init_strength = float(pipe._denoise_timesteps[0].item()) / 1000.0
    noise = pipe._sample_noise(input_latents.shape)
    noisy = input_latents * (1.0 - init_strength) + noise * init_strength
    context_tokens = pipe._fill_clean_context_cache(pipe._current_context_latents())
    denoised = pipe._denoise_latents(noisy, context_tokens)
    pipe._append_clean_latents(denoised)
    pipe._block_idx += 1
    return denoised, block_idx


class KreaOrchestratorStream(KreaOrchestrator):
    def _run_one_chunk(self):
        polled = self._poll_input_frames()
        if polled is None:
            time.sleep(0.002)
            return
        krea_input, raw_frames_np = polled
        n_pushed = int(raw_frames_np.shape[0])

        # SAM starts immediately (concurrent with denoise)
        self.sam_link.push_frames(raw_frames_np)
        mask_start = self._sam_push_count
        self._sam_push_count += n_pushed

        # sync: denoise returns a device tensor, so the work is still queued
        denoised, block_idx = denoise_chunk(self.pipe, krea_input)
        if denoised is None:
            return

        drop = self._output_frame_skip_frames  # 3 on chunk 0, else 0
        p = 0  # running pixel-frame index within this chunk (pre-drop)
        for frame_i in range(int(denoised.shape[2])):
            latent_i = denoised[:, :, frame_i:frame_i + 1, :, :].clone()
            is_first = (block_idx == 0 and frame_i == 0)
            decoded_i = self.pipe.vae_runner.run(latent_i, is_first)  # [B, Tf, H, W, 3]
            pix = np.asarray(decoded_i[0].cpu().numpy())              # [Tf, H, W, 3] uint8
            for j in range(int(pix.shape[0])):
                if p < drop:
                    p += 1
                    continue
                # output pixel p <-> raw frame p <-> mask[mask_start+p]
                orig = raw_frames_np[p]
                mask = self.sam_link.read_masks(mask_start + p, 1)[0]  # [H,W]
                out = self._composite_one(pix[j], orig, mask)
                self.video_buffer.write(out)
                p += 1
        # the causal-VAE warmup-frame drop only applies to chunk 0; p reached
        # n_pushed >= drop so it is fully consumed.
        self._output_frame_skip_frames = 0

    @staticmethod
    def _composite_one(krea_px, orig_px, mask):
        m = (mask > 0).astype(np.uint8)[:, :, None]
        return orig_px * m + krea_px * (1 - m)
