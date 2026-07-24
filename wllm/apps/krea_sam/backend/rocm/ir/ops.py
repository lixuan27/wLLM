"""IR operators for the Krea-Realtime + SAM3 pipeline.

Each operator faithfully re-expresses one stage of the reference backend
(wllm/apps/krea_sam/reference/{worker,pipeline}.py). Compute-heavy stages call the
*same* shared runners the reference uses (DiTRunner / VAERunner / SAM
predictor), so numerics match by construction; the light math (renoise, x0,
composite) is replicated exactly from the reference.

Cross-chunk state is threaded through the IR StateStore so the analysis tools
see the true dependency structure:
  * clean_latent_context  (chunk_persistent): DiT clean-context recurrence.
  * dit_kv_cache          (chunk_persistent): the DiT KV cache object every
    DiT forward this chunk reads/writes (refilled from clean_latent_context).
  * encoder_kv_cache      (session_init):     text cross-attn KV (filled once).
  * vae_encoder_cache     (chunk_persistent): streaming causal encode cache.
  * vae_decoder_cache     (chunk_persistent): causal decode cache.
  * sam_tracker_state     (chunk_persistent): SAM per-session tracking memory.

The `context` handed to execute() is an IRContext carrying the runners, the
SAM handle, per-chunk scalars (block_idx), and config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from wllm.serving.ir import IROperator, OpType, StreamMode, TensorPort, ir_operator


# ----------------------------------------------------------------------------
# context object
# ----------------------------------------------------------------------------

@dataclass
class IRContext:
    pipe: Any                     # KreaSAMPipeline (has vae_runner, dit_runner, math helpers)
    cfg: Any                      # RTConfig
    device: torch.device
    dtype: torch.dtype
    sam: Any = None               # SAM stream predictor (or None)
    sam_session_id: Optional[str] = None
    sam_prompt_set: bool = False
    sam_frame_index: int = 0
    block_idx: int = 0            # set by the driver before each run_chunk


# ----------------------------------------------------------------------------
# pure helpers copied verbatim from the reference pipeline (context management)
# ----------------------------------------------------------------------------

def current_context_latents(clean_ctx: Optional[torch.Tensor], cfg) -> Optional[torch.Tensor]:
    """Mirror KreaSAMPipeline._current_context_latents (pipeline.py:126-140)."""
    if clean_ctx is None or clean_ctx.shape[2] == 0:
        return None
    max_ctx = int(cfg.context_window_size)
    context = clean_ctx
    if context.shape[2] <= max_ctx:
        return context
    if cfg.keep_first_frame and max_ctx > 1:
        first = context[:, :, :1]
        tail = context[:, :, -max(0, max_ctx - 1):]
        return torch.cat([first, tail], dim=2)
    return context[:, :, -max_ctx:]


def append_clean_latents(clean_ctx: Optional[torch.Tensor], new_latents: torch.Tensor, cfg) -> torch.Tensor:
    """Mirror KreaSAMPipeline._append_clean_latents (pipeline.py:142-158)."""
    if clean_ctx is None:
        combined = new_latents.detach().clone()
    else:
        combined = torch.cat([clean_ctx, new_latents.detach()], dim=2)
    max_keep = int(cfg.context_window_size)
    if cfg.keep_first_frame and max_keep > 1 and combined.shape[2] > max_keep:
        combined = torch.cat([combined[:, :, :1], combined[:, :, -(max_keep - 1):]], dim=2)
    elif combined.shape[2] > max_keep:
        combined = combined[:, :, -max_keep:]
    return combined.contiguous()


# ----------------------------------------------------------------------------
# preamble: session init (seed, prompt encode, encoder-KV fill)
# ----------------------------------------------------------------------------

@ir_operator(
    name="session_init",
    op_type=OpType.EXPOSED,
    inputs=[TensorPort("prompt")],
    outputs=[TensorPort("ready")],
    state_writes=["encoder_kv_cache", "clean_latent_context"],
)
def session_init(inputs, ctx: IRContext, state):
    """Reset caches, seed noise, encode prompt, fill DiT encoder KV — mirrors
    the parts of worker.start()/pipe.init_session that set up a session."""
    from wllm.serving.utils.rand import set_global_seed
    set_global_seed(ctx.cfg.seed)
    ctx.pipe.init_session(prompt=inputs["prompt"], negative_prompt=ctx.cfg.negative_prompt or None)
    state.set("encoder_kv_cache", ctx.pipe.dit_runner.kv_memory)
    state.set("clean_latent_context", None)
    ctx.block_idx = 0
    return {"ready": True}


# ----------------------------------------------------------------------------
# chunk ops
# ----------------------------------------------------------------------------

class VaeEncode(IROperator):
    """Streaming causal VAE encode: pixel frames -> input latents.

    stream=False on chunk 0 (fresh causal encode), stream=True after (continues
    the encoder temporal cache — the chunk_persistent vae_encoder_cache).
    """

    def __init__(self):
        super().__init__(
            name="vae_encode",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("input_pixels")],
            outputs=[TensorPort("input_latents")],
            state_reads=["vae_encoder_cache"],
            state_writes=["vae_encoder_cache"],
            stream_mode=StreamMode.STREAMING,
        )

    def execute(self, inputs, ctx: IRContext, state):
        state.get("vae_encoder_cache")  # honor the declared in-place dep
        pipe = ctx.pipe
        input_frames = inputs["input_pixels"]  # [T, C, H, W] in [-1,1]
        pixels = input_frames.permute(1, 0, 2, 3).unsqueeze(0).to(device=ctx.device, dtype=ctx.dtype)
        input_latents = pipe.vae_runner.encode(pixels, stream=ctx.block_idx > 0).to(
            device=ctx.device, dtype=ctx.dtype)
        chunk_frames = int(ctx.cfg.chunk_size)
        if int(input_latents.shape[2]) < chunk_frames:
            return {"input_latents": None}
        input_latents = input_latents[:, :, -chunk_frames:].contiguous()
        return {"input_latents": input_latents}


@ir_operator(
    name="prepare_noisy",
    op_type=OpType.EXPOSED,
    inputs=[TensorPort("input_latents")],
    outputs=[TensorPort("noisy_latents")],
)
def prepare_noisy(inputs, ctx: IRContext, state):
    """Initial renoise: x_t = (1-s0)*x0 + s0*eps  (pipeline.py:384-386).

    Draws the FIRST of the 4 per-chunk noise samples (order matters for RNG
    parity with the reference)."""
    pipe = ctx.pipe
    input_latents = inputs["input_latents"]
    init_strength = float(pipe._denoise_timesteps[0].item()) / 1000.0
    noise = pipe._sample_noise(input_latents.shape)
    noisy = input_latents * (1.0 - init_strength) + noise * init_strength
    return {"noisy_latents": noisy}


class DitCacheFill(IROperator):
    """Fill the DiT KV cache with the clean-latent context (is_cache=True).

    Reads the chunk_persistent clean_latent_context (from the previous chunk),
    writes the context region of dit_kv_cache. Returns context_tokens=0 on
    chunk 0 (no context)."""

    def __init__(self):
        super().__init__(
            name="dit_cache_fill",
            op_type=OpType.EXPOSED,
            inputs=[],
            outputs=[TensorPort("context_tokens")],
            state_reads=["clean_latent_context", "encoder_kv_cache"],
            state_writes=["dit_kv_cache"],
        )

    def execute(self, inputs, ctx: IRContext, state):
        state.get("encoder_kv_cache")
        clean_ctx = state.get("clean_latent_context")
        context = current_context_latents(clean_ctx, ctx.cfg)
        context_tokens = ctx.pipe._fill_clean_context_cache(context)
        return {"context_tokens": int(context_tokens)}


class DitDenoiseStep(IROperator):
    """One flow-matching denoising step k (of num_inference_steps).

    Each step runs a DiT forward (is_cache=False), converts velocity->x0, and
    (for k < last) renoises to the next timestep — drawing renoise noise in the
    exact order the reference does. Reads/writes the dit_kv_cache (chunk region)
    and reads encoder_kv_cache (cross-attn)."""

    def __init__(self, step_idx: int, num_steps: int):
        self.step_idx = step_idx
        self.num_steps = num_steps
        is_last = step_idx == num_steps - 1
        super().__init__(
            name=f"dit_denoise_{step_idx}",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("latents_in"), TensorPort("context_tokens")],
            outputs=[TensorPort("denoised" if is_last else "latents_out")],
            state_reads=["dit_kv_cache", "encoder_kv_cache"],
            state_writes=["dit_kv_cache"],
        )
        self._out_port = "denoised" if is_last else "latents_out"

    def execute(self, inputs, ctx: IRContext, state):
        state.get("dit_kv_cache")
        state.get("encoder_kv_cache")
        pipe = ctx.pipe
        current = inputs["latents_in"].to(device=ctx.device, dtype=ctx.dtype)
        context_tokens = int(inputs["context_tokens"])
        chunk_frames = int(current.shape[2])
        chunk_tokens = chunk_frames * int(ctx.cfg.kv_spatial)

        timestep_value = pipe._denoise_timesteps[self.step_idx]
        timestep = torch.full((chunk_frames,), timestep_value, device=ctx.device,
                              dtype=pipe._denoise_timesteps.dtype)
        flow_pred = pipe.dit_runner.run(
            latents=current, timestep=timestep, is_cache=False,
            cache_start=context_tokens, cache_end=context_tokens + chunk_tokens,
            rope_start=context_tokens, rope_end=context_tokens + chunk_tokens,
        )
        sigma_t = timestep_value.to(torch.float64) / 1000.0
        x0_pred = (current.to(torch.float64) - sigma_t * flow_pred.to(torch.float64)).to(current.dtype)
        if self.step_idx < self.num_steps - 1:
            out = pipe._renoise(x0_pred, pipe._denoise_timesteps[self.step_idx + 1])
        else:
            out = x0_pred
        return {self._out_port: out}


class DitAppendContext(IROperator):
    """Append this chunk's denoised latents to the clean-latent context
    (chunk_persistent recurrence that conditions the next chunk's DiT)."""

    def __init__(self):
        super().__init__(
            name="dit_append_context",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("denoised")],
            outputs=[TensorPort("denoised_out")],
            state_reads=["clean_latent_context"],
            state_writes=["clean_latent_context"],
        )

    def execute(self, inputs, ctx: IRContext, state):
        denoised = inputs["denoised"]
        clean_ctx = state.get("clean_latent_context")
        new_ctx = append_clean_latents(clean_ctx, denoised, ctx.cfg)
        state.set("clean_latent_context", new_ctx)
        return {"denoised_out": denoised}


class VaeDecode(IROperator):
    """Causal VAE decode of the chunk's latent frames -> uint8 RGB pixels.

    Per-frame decode loop (as the reference), threading the chunk_persistent
    vae_decoder_cache. is_first only on (block 0, frame 0)."""

    def __init__(self):
        super().__init__(
            name="vae_decode",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("denoised")],
            outputs=[TensorPort("krea_frames")],
            state_reads=["vae_decoder_cache"],
            state_writes=["vae_decoder_cache"],
            stream_mode=StreamMode.STREAMING,
        )

    def execute(self, inputs, ctx: IRContext, state):
        state.get("vae_decoder_cache")
        pipe = ctx.pipe
        denoised = inputs["denoised"]
        chunk_video = []
        for frame_i in range(int(denoised.shape[2])):
            latent_i = denoised[:, :, frame_i:frame_i + 1, :, :].clone()
            is_first = (ctx.block_idx == 0 and frame_i == 0)
            decoded_i = pipe.vae_runner.run(latent_i, is_first)
            chunk_video.append(decoded_i[0].cpu().numpy())
        return {"krea_frames": np.concatenate(chunk_video, axis=0)}


class SamSegment(IROperator):
    """SAM3 person segmentation over the chunk's raw frames (BLACK_BOX).

    Sequential per-frame inference threading the chunk_persistent
    sam_tracker_state. Mirrors worker._run_sam exactly."""

    def __init__(self):
        super().__init__(
            name="sam_segment",
            op_type=OpType.BLACK_BOX,
            inputs=[TensorPort("raw_frames")],
            outputs=[TensorPort("masks")],
            state_reads=["sam_tracker_state"],
            state_writes=["sam_tracker_state"],
        )

    def execute(self, inputs, ctx: IRContext, state):
        state.get("sam_tracker_state")
        frames_np = inputs["raw_frames"]
        masks = _run_sam(ctx, frames_np)
        return {"masks": masks}


class Composite(IROperator):
    """Drop first-chunk VAE warmup frames, then composite: SAM-body pixels keep
    the original webcam frame, background keeps the Krea-stylised frame.
    Mirrors the tail of worker.loop()."""

    def __init__(self):
        super().__init__(
            name="composite",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("krea_frames"), TensorPort("raw_frames"), TensorPort("masks")],
            outputs=[TensorPort("composited")],
        )

    def execute(self, inputs, ctx: IRContext, state):
        krea_frames = inputs["krea_frames"]
        raw_frames_np = inputs["raw_frames"]
        masks = inputs["masks"]

        # first-chunk causal-VAE warmup-frame drop (worker.loop:408-416)
        skip_frames = max(0, int(ctx.cfg.vae_config.scale_factor_temporal) - 1) if ctx.block_idx == 0 else 0
        if skip_frames > 0:
            skip = min(skip_frames, int(krea_frames.shape[0]))
            krea_frames = krea_frames[skip:]
            raw_frames_np = raw_frames_np[skip:]
            if masks is not None:
                masks = masks[skip:]
            if krea_frames.shape[0] == 0:
                return {"composited": krea_frames}

        n_out = int(krea_frames.shape[0])
        originals = raw_frames_np[:n_out] if raw_frames_np.shape[0] >= n_out else None
        if originals is None or originals.shape[:3] != krea_frames.shape[:3]:
            return {"composited": krea_frames}
        if masks is not None and masks.shape[0] >= n_out:
            masks = masks[:n_out]
        else:
            masks = None
        return {"composited": _composite(krea_frames, originals, masks)}


# ----------------------------------------------------------------------------
# SAM + composite helpers, copied from worker.py to preserve exact behavior
# ----------------------------------------------------------------------------

def _run_sam(ctx: IRContext, frames_np: np.ndarray) -> Optional[np.ndarray]:
    """Verbatim port of KreaSAMWorker._run_sam (worker.py:201-290)."""
    if ctx.sam is None or ctx.sam_session_id is None:
        return None
    T, H, W, _ = frames_np.shape
    out = np.zeros((T, H, W), dtype=np.uint8)
    score_thresh = float(ctx.cfg.sam_min_score)
    mask_thresh = float(ctx.cfg.sam_mask_threshold)
    dilate_px = int(ctx.cfg.sam_dilate_pixels)
    cv2 = None
    for i in range(T):
        ctx.sam.handle_request({"type": "add_frame", "session_id": ctx.sam_session_id, "frame": frames_np[i]})
        if not ctx.sam_prompt_set:
            resp = ctx.sam.handle_request({"type": "add_prompt", "session_id": ctx.sam_session_id,
                                           "frame_index": ctx.sam_frame_index, "text": ctx.cfg.sam_text_prompt})
            ctx.sam_prompt_set = True
        else:
            resp = ctx.sam.handle_request({"type": "run_inference", "session_id": ctx.sam_session_id,
                                           "frame_index": ctx.sam_frame_index})
        ctx.sam_frame_index += 1

        outputs = (resp or {}).get("outputs") or {}
        raw_masks = outputs.get("out_binary_masks")
        raw_probs = outputs.get("out_probs")
        masks = list(raw_masks) if raw_masks is not None and len(raw_masks) > 0 else []
        probs = list(raw_probs) if raw_probs is not None and len(raw_probs) > 0 else []
        if not masks or not probs:
            continue
        mask_union = np.zeros((H, W), dtype=bool)
        for m, s in zip(masks, probs):
            try:
                score = float(s)
            except (TypeError, ValueError):
                score = 0.0
            if score < score_thresh:
                continue
            m_np = np.asarray(m)
            if m_np.ndim == 3 and m_np.shape[0] == 1:
                m_np = m_np[0]
            if m_np.shape != (H, W):
                if cv2 is None:
                    import cv2 as _cv2
                    cv2 = _cv2
                m_np = cv2.resize(m_np.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            mask_union |= (m_np > mask_thresh)
        if dilate_px > 0:
            if cv2 is None:
                import cv2 as _cv2
                cv2 = _cv2
            k = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), dtype=np.uint8)
            mask_union = cv2.dilate(mask_union.astype(np.uint8), k) > 0
        out[i] = mask_union.astype(np.uint8) * 255
    return out


def _composite(krea_frames: np.ndarray, original_frames: np.ndarray,
               masks: Optional[np.ndarray]) -> np.ndarray:
    """Verbatim port of KreaSAMWorker._composite (worker.py:292-303)."""
    if masks is None:
        return krea_frames
    m3 = (masks > 0).astype(np.uint8)[:, :, :, None]
    return original_frames * m3 + krea_frames * (1 - m3)
