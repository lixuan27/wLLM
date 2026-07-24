"""Streaming variant of KreaSAMPipeline.step.

Vendored from `wllm/apps/krea_sam/reference/pipeline.py::KreaSAMPipeline.step`
(read-only) so the *math* is byte-identical to the reference, but the
per-latent-frame VAE decode emits its frames through a callback as soon
as each frame is decoded, instead of accumulating the whole chunk. This
realizes the model-graph streaming edge `vae_decode → composite` on the
producer side (the Krea service), so the compositor can start emitting
before the chunk finishes decoding.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch


@torch.inference_mode()
def streaming_step(pipe, input_frames: torch.Tensor, block_idx: int,
                   on_decoded: Callable[[np.ndarray], None]) -> bool:
    """Run one v2v chunk, calling ``on_decoded(frames_np)`` after each
    latent frame is decoded. Returns True if a chunk was produced, False
    if the streaming encoder is still priming.
    """
    pipe._block_idx = block_idx
    pixels = input_frames.permute(1, 0, 2, 3).unsqueeze(0).to(device=pipe.device, dtype=pipe.dtype)
    input_latents = pipe.vae_runner.encode(pixels, stream=block_idx > 0).to(
        device=pipe.device, dtype=pipe.dtype)
    chunk_frames = int(pipe.cfg.chunk_size)
    if int(input_latents.shape[2]) < chunk_frames:
        return False
    input_latents = input_latents[:, :, -chunk_frames:].contiguous()

    init_strength = float(pipe._denoise_timesteps[0].item()) / 1000.0
    noise = pipe._sample_noise(input_latents.shape)
    noisy_latents = input_latents * (1.0 - init_strength) + noise * init_strength

    clean_context = pipe._current_context_latents()
    context_tokens = pipe._fill_clean_context_cache(clean_context)
    denoised_latents = pipe._denoise_latents(noisy_latents, context_tokens)
    pipe._append_clean_latents(denoised_latents)

    for frame_i in range(int(denoised_latents.shape[2])):
        latent_i = denoised_latents[:, :, frame_i:frame_i + 1, :, :].clone()
        is_first = (block_idx == 0 and frame_i == 0)
        decoded_i = pipe.vae_runner.run(latent_i, is_first)
        on_decoded(decoded_i[0].cpu().numpy())  # [T_i, H, W, 3] uint8
    return True
