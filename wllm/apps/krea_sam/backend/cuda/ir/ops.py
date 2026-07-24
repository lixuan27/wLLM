"""IR operators for the Krea+SAM pipeline.

Two op families:

* **Krea model-level ops** decompose ``KreaSAMPipeline.step`` into the
  fine-grained operations whose state independence the analysis tools
  read: streaming VAE encode, init-noise blend, clean-context KV fill,
  the N denoising steps, clean-context append, and per-frame VAE decode.
  Each calls the *real* DiT / VAE runner so the executor runs the real
  computation. Logic is ported verbatim from the reference pipeline so
  the IR is byte-faithful.

* **Worker-level ops** model the high-level schedule: ``krea_v2v``
  (a COMPOSITE wrapping the Krea model sub-graph), ``sam_segment`` (the
  BLACK_BOX SAM stream model), and ``composite`` (background swap). The
  worker graph makes the key structural fact explicit: ``krea_v2v`` and
  ``sam_segment`` consume the same input frames and share **no** state,
  so they are independent and can run concurrently.

State objects (declared in graph_builder, referenced here):
  clean_latent_context  - denoised-latent context, conditions the DiT
  dit_kv                - DiT prefix+chunk KV cache (in dit_runner)
  vae_enc_cache         - causal temporal cache of the streaming encoder
  vae_dec_cache         - causal temporal cache of the decoder
  sam_tracking          - SAM per-session tracking memory
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from wllm.serving.ir import IROperator, OpType, StreamMode, TensorPort, ir_operator


# ----------------------------------------------------------------------
# helpers ported from KreaSAMPipeline (kept identical for fidelity)
# ----------------------------------------------------------------------

def _window_context(context: Optional[torch.Tensor], cfg) -> Optional[torch.Tensor]:
    """Replicates KreaSAMPipeline._current_context_latents."""
    if context is None or context.shape[2] == 0:
        return None
    max_ctx = int(cfg.context_window_size)
    if context.shape[2] <= max_ctx:
        return context
    if cfg.keep_first_frame and max_ctx > 1:
        first = context[:, :, :1]
        tail = context[:, :, -max(0, max_ctx - 1):]
        return torch.cat([first, tail], dim=2)
    return context[:, :, -max_ctx:]


def _append_context(prev: Optional[torch.Tensor], new_latents: torch.Tensor, cfg) -> torch.Tensor:
    """Replicates KreaSAMPipeline._append_clean_latents."""
    if prev is None:
        combined = new_latents.detach().clone()
    else:
        combined = torch.cat([prev, new_latents.detach()], dim=2)
    max_keep = int(cfg.context_window_size)
    if cfg.keep_first_frame and max_keep > 1 and combined.shape[2] > max_keep:
        combined = torch.cat([combined[:, :, :1], combined[:, :, -(max_keep - 1):]], dim=2)
    elif combined.shape[2] > max_keep:
        combined = combined[:, :, -max_keep:]
    return combined.contiguous()


# ----------------------------------------------------------------------
# Krea model-level ops
# ----------------------------------------------------------------------

@ir_operator(
    name="vae_encode",
    op_type=OpType.EXPOSED,
    inputs=[TensorPort("input_frames", ("T", "C", "H", "W"))],
    outputs=[TensorPort("input_latents", ("1", "z", "Tc", "h", "w"))],
    state_reads=["vae_enc_cache"],
    state_writes=["vae_enc_cache"],
    stream_mode=StreamMode.STREAMING,
)
def vae_encode(inputs, ctx, state):
    state.get("vae_enc_cache")  # declare dependency on the streaming cache
    pipe = ctx.pipe
    cfg = ctx.cfg
    frames = inputs["input_frames"]
    pixels = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device=ctx.device, dtype=pipe.dtype)
    input_latents = pipe.vae_runner.encode(pixels, stream=ctx.block_idx > 0).to(
        device=ctx.device, dtype=pipe.dtype)
    chunk_frames = int(cfg.chunk_size)
    if int(input_latents.shape[2]) < chunk_frames:
        # streaming encoder still priming; downstream ops are skipped this chunk
        return {"input_latents": None}
    input_latents = input_latents[:, :, -chunk_frames:].contiguous()
    return {"input_latents": input_latents}


@ir_operator(
    name="add_noise",
    op_type=OpType.EXPOSED,
    inputs=[TensorPort("input_latents")],
    outputs=[TensorPort("noisy_latents")],
)
def add_noise(inputs, ctx, state):
    pipe = ctx.pipe
    input_latents = inputs["input_latents"]
    init_strength = float(pipe._denoise_timesteps[0].item()) / 1000.0
    noise = pipe._sample_noise(input_latents.shape)
    noisy = input_latents * (1.0 - init_strength) + noise * init_strength
    return {"noisy_latents": noisy}


@ir_operator(
    name="fill_context",
    op_type=OpType.EXPOSED,
    inputs=[TensorPort("input_latents")],  # ordering anchor only
    outputs=[TensorPort("context_tokens")],
    state_reads=["clean_latent_context"],
    state_writes=["dit_kv"],
)
def fill_context(inputs, ctx, state):
    del inputs
    state.get("dit_kv")  # this op writes the prefix region of the KV cache
    clean = _window_context(state.get("clean_latent_context"), ctx.cfg)
    context_tokens = ctx.pipe._fill_clean_context_cache(clean)
    return {"context_tokens": context_tokens}


class DiTDenoiseStep(IROperator):
    """One flow-matching denoising step (DiT forward + Euler-blend renoise).

    Instantiated once per inference step. All steps write the same chunk
    region of the DiT KV cache (``dit_kv``) and chain through the latent
    data edge, so they form one serial pipeline stage — the DiT.
    """

    def __init__(self, step_idx: int, num_steps: int):
        super().__init__(
            name=f"denoise_step_{step_idx}",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("latents_in"), TensorPort("context_tokens")],
            outputs=[TensorPort("latents_out")],
            state_reads=["dit_kv"],
            state_writes=["dit_kv"],
        )
        self.step_idx = step_idx
        self.num_steps = num_steps

    def execute(self, inputs, ctx, state):
        state.get("dit_kv")
        pipe = ctx.pipe
        cfg = ctx.cfg
        current = inputs["latents_in"].to(device=ctx.device, dtype=pipe.dtype)
        context_tokens = int(inputs["context_tokens"])
        chunk_frames = int(current.shape[2])
        chunk_tokens = chunk_frames * int(cfg.kv_spatial)

        timestep_value = pipe._denoise_timesteps[self.step_idx]
        timestep = torch.full((chunk_frames,), timestep_value,
                              device=ctx.device, dtype=pipe._denoise_timesteps.dtype)
        flow_pred = pipe.dit_runner.run(
            latents=current, timestep=timestep, is_cache=False,
            cache_start=context_tokens, cache_end=context_tokens + chunk_tokens,
            rope_start=context_tokens, rope_end=context_tokens + chunk_tokens,
        )
        sigma_t = timestep_value.to(torch.float64) / 1000.0
        x0_pred = (current.to(torch.float64) - sigma_t * flow_pred.to(torch.float64)).to(current.dtype)
        if self.step_idx < (self.num_steps - 1):
            out = pipe._renoise(x0_pred, pipe._denoise_timesteps[self.step_idx + 1])
        else:
            out = x0_pred
        return {"latents_out": out}


@ir_operator(
    name="append_context",
    op_type=OpType.EXPOSED,
    inputs=[TensorPort("denoised_latents")],
    outputs=[TensorPort("denoised_passthrough")],
    state_reads=["clean_latent_context"],
    state_writes=["clean_latent_context"],
)
def append_context(inputs, ctx, state):
    denoised = inputs["denoised_latents"]
    combined = _append_context(state.get("clean_latent_context"), denoised, ctx.cfg)
    state.set("clean_latent_context", combined)
    return {"denoised_passthrough": denoised}


@ir_operator(
    name="vae_decode",
    op_type=OpType.EXPOSED,
    inputs=[TensorPort("denoised_latents")],
    outputs=[TensorPort("krea_frames", ("T", "H", "W", "3"))],
    state_reads=["vae_dec_cache"],
    state_writes=["vae_dec_cache"],
    stream_mode=StreamMode.STREAMING,
)
def vae_decode(inputs, ctx, state):
    state.get("vae_dec_cache")
    pipe = ctx.pipe
    denoised = inputs["denoised_latents"]
    chunk_video: list[np.ndarray] = []
    for frame_i in range(int(denoised.shape[2])):
        latent_i = denoised[:, :, frame_i:frame_i + 1, :, :].clone()
        is_first = (ctx.block_idx == 0 and frame_i == 0)
        decoded_i = pipe.vae_runner.run(latent_i, is_first)
        chunk_video.append(decoded_i[0].cpu().numpy())
    return {"krea_frames": np.concatenate(chunk_video, axis=0)}


# ----------------------------------------------------------------------
# Worker-level ops
# ----------------------------------------------------------------------

@ir_operator(
    name="krea_v2v",
    op_type=OpType.COMPOSITE,
    inputs=[TensorPort("input_frames")],
    outputs=[TensorPort("krea_frames")],
    state_reads=["clean_latent_context", "vae_enc_cache", "vae_dec_cache", "dit_kv"],
    state_writes=["clean_latent_context", "vae_enc_cache", "vae_dec_cache", "dit_kv"],
    sub_graph="krea_model",
    stream_mode=StreamMode.STREAMING,
)
def krea_v2v(inputs, ctx, state):
    """Run the Krea v2v model sub-graph for one chunk via the nested
    executor held on the context. Returns the decoded (pre-composite)
    chunk frames, or None while the streaming encoder primes."""
    out = ctx.krea_executor.run_chunk({"input_frames": inputs["input_frames"]}, ctx)
    return {"krea_frames": out.get("krea_frames")}


@ir_operator(
    name="sam_segment",
    op_type=OpType.BLACK_BOX,
    inputs=[TensorPort("raw_frames", ("T", "H", "W", "3"))],
    outputs=[TensorPort("masks", ("T", "H", "W"))],
    state_reads=["sam_tracking"],
    state_writes=["sam_tracking"],
    stream_mode=StreamMode.STREAMING,
)
def sam_segment(inputs, ctx, state):
    state.get("sam_tracking")
    masks = ctx.run_sam(inputs["raw_frames"])
    return {"masks": masks}


@ir_operator(
    name="composite",
    op_type=OpType.EXPOSED,
    inputs=[TensorPort("krea_frames"), TensorPort("raw_frames"), TensorPort("masks")],
    outputs=[TensorPort("composited", ("T", "H", "W", "3"))],
)
def composite(inputs, ctx, state):
    """Background swap + session-warmup frame drop (vendored from worker.loop)."""
    krea_frames = inputs["krea_frames"]
    raw_frames_np = inputs["raw_frames"]
    masks = inputs["masks"]

    if krea_frames is None or len(krea_frames) == 0:
        return {"composited": None}

    if ctx._output_frame_skip_frames > 0:
        skip = min(ctx._output_frame_skip_frames, int(krea_frames.shape[0]))
        krea_frames = krea_frames[skip:]
        raw_frames_np = raw_frames_np[skip:]
        if masks is not None:
            masks = masks[skip:]
        ctx._output_frame_skip_frames -= skip
        if krea_frames.shape[0] == 0:
            return {"composited": None}

    n_out = int(krea_frames.shape[0])
    originals = raw_frames_np[:n_out] if raw_frames_np.shape[0] >= n_out else None
    if originals is None or originals.shape[:3] != krea_frames.shape[:3]:
        return {"composited": krea_frames}
    if masks is not None and masks.shape[0] >= n_out:
        masks = masks[:n_out]
    else:
        masks = None
    return {"composited": ctx.composite(krea_frames, originals, masks)}
