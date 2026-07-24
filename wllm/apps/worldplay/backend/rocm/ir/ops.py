"""IR operator definitions for the WorldPlay chunk graph.

Each operator wraps one slice of the reference chunk (see
`pipeline_decomposed.WorldPlayDecomposedPipeline`). Persistent tensor state
(latent store, KV cache, VAE temporal cache, conditioning accumulators, camera
pose) is declared via `state_reads` / `state_writes` so the analysis tools can
derive the DiT-vs-VAE pipeline split and the cross-chunk independent pairs;
the tensors themselves live on `ctx.pipe` and the finalized chunk latents are
handed to the VAE as a **data edge** (not shared state), which is what makes
the VAE stage independent of the next chunk's DiT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from wllm.serving.ir import IROperator, OpType, StreamMode, TensorPort, ir_operator

from wllm.apps.worldplay.backend.rocm.ir.pipeline_decomposed import WorldPlayDecomposedPipeline


@dataclass
class WorldPlayCtx:
    """Execution context handed to every operator (opaque to the executor)."""
    pipe: WorldPlayDecomposedPipeline
    chunk_idx: int = 0            # current latent_chunk_idx (before advance)
    prompt: str = ""
    image_path: str = ""


# --- state object names (declared in graph_builder) -----------------------
S_CAMERA = "camera_pose"          # T / C_inv accumulators
S_COND = "cond_accum"             # viewmats / Ks / action accumulators
S_LATENTS = "latents"             # fp32 latent store (cross-chunk context)
S_KV = "kv_cache"                 # DiT KV cache (prope)
S_VAE = "vae_cache"               # VAE temporal causal cache


# ==========================================================================
# preamble: session init (text encode + image encode + encoder-KV prefill)
# ==========================================================================

@ir_operator(
    name="session_init",
    op_type=OpType.EXPOSED,
    inputs=[],
    outputs=[],
    state_reads=[],
    state_writes=[S_LATENTS, S_KV],
)
def session_init(inputs, ctx, state):
    # encodes the prompt (T5), encodes the first image (VAE), fills the DiT
    # cross-attention encoder KV, and prepares the noise latents.
    ctx.pipe.init_session(prompt=ctx.prompt, negative_prompt=None,
                          image_path=ctx.image_path)
    return {}


# ==========================================================================
# camera + conditioning
# ==========================================================================

@ir_operator(
    name="camera_decode",
    op_type=OpType.EXPOSED,
    inputs=[TensorPort("actions")],
    outputs=[TensorPort("viewmats"), TensorPort("Ks"), TensorPort("action")],
    state_reads=[S_CAMERA],
    state_writes=[S_CAMERA],
)
def camera_decode(inputs, ctx, state):
    _ = state.get(S_CAMERA)   # T/C_inv live on ctx.pipe; declared for analysis
    viewmats, Ks, action = ctx.pipe.op_camera_decode(inputs["actions"])
    return {"viewmats": viewmats, "Ks": Ks, "action": action}


@ir_operator(
    name="prep",
    op_type=OpType.EXPOSED,
    inputs=[TensorPort("viewmats"), TensorPort("Ks"), TensorPort("action")],
    outputs=[],
    state_reads=[S_LATENTS, S_COND],
    state_writes=[S_LATENTS, S_COND],
)
def prep(inputs, ctx, state):
    _ = state.get(S_LATENTS)
    ctx.pipe.op_prep(inputs["viewmats"], inputs["Ks"], inputs["action"])
    return {}


def _chunk_gt0(ctx) -> bool:
    return ctx.chunk_idx > 0


@ir_operator(
    name="select_mem",
    op_type=OpType.EXPOSED,
    inputs=[],
    outputs=[],
    state_reads=[S_COND],
    state_writes=[],
    should_run=_chunk_gt0,
)
def select_mem(inputs, ctx, state):
    _ = state.get(S_COND)
    ctx.pipe.op_select_mem()
    return {}


@ir_operator(
    name="kv_fill",
    op_type=OpType.EXPOSED,
    inputs=[],
    outputs=[],
    state_reads=[S_LATENTS, S_COND],
    state_writes=[S_KV],
    should_run=_chunk_gt0,
)
def kv_fill(inputs, ctx, state):
    _ = state.get(S_LATENTS)
    ctx.pipe.op_kv_fill()
    return {}


class DenoiseStep(IROperator):
    """One denoising step i: gen forward (attends to KV cache) + Euler update.

    Instantiated once per inference step; each reads/writes the latent store
    and the KV cache, so all steps are chained into the DiT pipeline stage.
    """

    def __init__(self, step_idx: int):
        super().__init__(
            name=f"denoise_{step_idx}",
            op_type=OpType.EXPOSED,
            inputs=[],
            outputs=[],
            state_reads=[S_LATENTS, S_KV, S_COND],
            state_writes=[S_LATENTS, S_KV],
        )
        self.step_idx = step_idx

    def execute(self, inputs, ctx, state):
        _ = state.get(S_KV)
        ctx.pipe.op_denoise(self.step_idx)
        return {}


@ir_operator(
    name="finalize",
    op_type=OpType.EXPOSED,
    inputs=[],
    outputs=[TensorPort("chunk_latents")],
    state_reads=[S_LATENTS],
    state_writes=[S_LATENTS],
)
def finalize(inputs, ctx, state):
    _ = state.get(S_LATENTS)
    start_idx, end_idx = ctx.pipe.op_finalize()
    # hand the finalized latents to the VAE as data (decoupled from the
    # cross-chunk `latents` state, which the next chunk's DiT keeps mutating)
    chunk_latents = ctx.pipe._latents[:, :, start_idx:end_idx, :, :].clone()
    return {"chunk_latents": chunk_latents}


# ==========================================================================
# VAE decode (own pipeline stage: shares only the VAE temporal cache)
# ==========================================================================

class VaeDecode(IROperator):
    """Decode one latent frame of the chunk to pixels. The 4 decodes chain
    through the VAE temporal causal cache (S_VAE), forming the VAE stage that
    is independent of the DiT stage and can overlap it across chunks."""

    def __init__(self, j: int):
        super().__init__(
            name=f"vae_decode_{j}",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("chunk_latents")],
            outputs=[TensorPort(f"frames_{j}")],
            state_reads=[S_VAE],
            state_writes=[S_VAE],
            stream_mode=StreamMode.STREAMING,
        )
        self.j = j

    def execute(self, inputs, ctx, state):
        _ = state.get(S_VAE)
        start_idx = ctx.pipe._scratch["start_idx"]
        l_i = start_idx + self.j
        frames = ctx.pipe.op_vae_decode(l_i)
        return {f"frames_{self.j}": frames}


class CollectFrames(IROperator):
    """Concatenate the per-latent pixel frames into the chunk's frame block
    and advance the chunk counter (mirrors the end of pipeline.step)."""

    def __init__(self, n: int):
        super().__init__(
            name="collect_frames",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort(f"frames_{j}") for j in range(n)],
            outputs=[TensorPort("chunk_frames")],
            state_reads=[],
            state_writes=[],
            stream_mode=StreamMode.STREAMING,
        )
        self.n = n

    def execute(self, inputs, ctx, state):
        frames = np.concatenate([inputs[f"frames_{j}"] for j in range(self.n)], axis=0)
        ctx.pipe.op_advance_chunk()
        return {"chunk_frames": frames}
