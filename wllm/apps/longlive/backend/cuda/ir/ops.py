"""IR operators for the LongLive pipeline.

Two families:
  * Model-level ops (executable) — the fine-grained decomposition of one
    ``LongLivePipeline.step()`` into noise / denoise / cache / VAE-decode
    operators. These call the shared ``backend/generation.py`` primitives so
    they are numerically identical to the reference. The SequentialExecutor
    runs these in Phase 2 validation.
  * Worker-level ops (structural) — high-level stages (audio/VAD, ASR
    black box, prompt encode, DiT chunk, VAE chunk) used for analysis of the
    pipeline-scheduling layer. Not executed.
"""
from __future__ import annotations

from typing import Any

from wllm.serving.ir import IROperator, OpType, TensorPort, StreamMode
from wllm.apps.longlive.backend.cuda import generation as G


# ===========================================================================
# Model-level executable operators
# ===========================================================================

class EncodePrompt(IROperator):
    """Preamble = reference init_session: seed the noise RNG, then UMT5
    text-encode + fill the DiT cross-attention KV (exposed). Seeding here
    mirrors ``LongLivePipeline.init_session``'s manual_seed; without it the
    chunk noise diverges from the reference."""

    def __init__(self):
        super().__init__(
            name="encode_prompt", op_type=OpType.EXPOSED,
            inputs=[TensorPort("prompt")], outputs=[TensorPort("encoder_kv")],
            state_writes=["encoder_kv"],
        )

    def execute(self, inputs, ctx, state):
        prompt = inputs["prompt"]
        ctx.core.seed()           # manual_seed(cfg.seed) — required for parity
        ctx.core.set_prompt(prompt)
        state.set("encoder_kv", prompt)
        return {"encoder_kv": prompt}


class ChunkPlan(IROperator):
    """Compute the KV write slot + rope window for this chunk from ring_state."""

    def __init__(self):
        super().__init__(
            name="chunk_plan", op_type=OpType.EXPOSED,
            inputs=[], outputs=[TensorPort("plan")],
            state_reads=["ring_state"],
        )

    def execute(self, inputs, ctx, state):
        ring = state.get("ring_state")
        return {"plan": G.plan_chunk(ctx.core, ring)}


class NoiseSample(IROperator):
    """Sample the chunk's initial latent noise (seeded RNG)."""

    def __init__(self):
        super().__init__(
            name="noise_sample", op_type=OpType.EXPOSED,
            inputs=[], outputs=[TensorPort("latents")],
        )

    def execute(self, inputs, ctx, state):
        return {"latents": G.initial_noise(ctx.core)}


class DenoiseStep(IROperator):
    """One DMD denoise step (is_cache=False DiT forward + fp64 x0 + renoise)."""

    def __init__(self, step_idx: int):
        super().__init__(
            name=f"denoise_{step_idx}", op_type=OpType.EXPOSED,
            inputs=[TensorPort("latents"), TensorPort("plan")],
            outputs=[TensorPort("latents")],
            state_reads=["kv_ring", "encoder_kv"], state_writes=["kv_ring"],
        )
        self.step_idx = step_idx

    def execute(self, inputs, ctx, state):
        state.get("kv_ring"); state.get("encoder_kv")
        out = G.denoise_one_step(ctx.core, inputs["latents"], inputs["plan"],
                                 self.step_idx)
        return {"latents": out}


class CacheWrite(IROperator):
    """t=0 pass: write clean K/V into the ring + advance ring bookkeeping."""

    def __init__(self):
        super().__init__(
            name="cache_write", op_type=OpType.EXPOSED,
            inputs=[TensorPort("latents"), TensorPort("plan")],
            outputs=[TensorPort("latents")],
            state_reads=["kv_ring", "encoder_kv", "ring_state"],
            state_writes=["kv_ring", "ring_state"],
        )

    def execute(self, inputs, ctx, state):
        state.get("kv_ring"); state.get("encoder_kv")
        ring = state.get("ring_state")
        G.write_clean_cache(ctx.core, inputs["latents"], inputs["plan"])
        # ring_state advance touches only block_idx/rolling/pinned/max_filled,
        # none of which the VAE ops read, so advancing here (before decode) is
        # numerically identical to the reference's advance-after-decode order.
        G.advance_ring(ctx.core, ring, inputs["plan"])
        return {"latents": inputs["latents"]}


class VaeDecode(IROperator):
    """Decode one latent frame to pixels (Wan causal streaming decode)."""

    def __init__(self, l: int):
        super().__init__(
            name=f"vae_decode_{l}", op_type=OpType.EXPOSED,
            inputs=[TensorPort("latents")], outputs=[TensorPort("frame")],
            state_reads=["vae_cache"], state_writes=["vae_cache", "video_out"],
            stream_mode=StreamMode.STREAMING,
        )
        self.l = l

    def execute(self, inputs, ctx, state):
        vc = state.get("vae_cache")
        is_first = (vc["count"] == 0)
        frame = G.decode_latent_frame(ctx.core, inputs["latents"], self.l, is_first)
        vc["count"] += 1
        state.get("video_out").append(frame)
        return {"frame": frame}


# ===========================================================================
# Worker-level structural operators (analysis only; not executed)
# ===========================================================================

def _structural(name, op_type, inputs, outputs, state_reads=None,
                state_writes=None, stream=StreamMode.BATCH):
    return IROperator(
        name=name, op_type=op_type,
        inputs=[TensorPort(p) for p in inputs],
        outputs=[TensorPort(p) for p in outputs],
        state_reads=state_reads or [], state_writes=state_writes or [],
        stream_mode=stream,
    )


def audio_vad_op():
    return _structural("audio_vad", OpType.EXPOSED, ["audio_chunk"],
                       ["utterance"], state_reads=["vad_state"],
                       state_writes=["vad_state"], stream=StreamMode.STREAMING)


def asr_op():
    return _structural("asr", OpType.BLACK_BOX, ["utterance"], ["prompt_text"],
                       stream=StreamMode.BATCH)


def apply_prompt_op():
    return _structural("apply_prompt", OpType.EXPOSED, ["prompt_text"],
                       ["encoder_kv"], state_writes=["encoder_kv"])


def dit_chunk_op():
    return IROperator(
        name="dit_chunk", op_type=OpType.COMPOSITE,
        inputs=[TensorPort("encoder_kv")], outputs=[TensorPort("latents")],
        state_reads=["kv_ring", "ring_state", "encoder_kv"],
        state_writes=["kv_ring", "ring_state"],
        sub_graph="longlive_model", stream_mode=StreamMode.STREAMING,
    )


def vae_chunk_op():
    return IROperator(
        name="vae_chunk", op_type=OpType.COMPOSITE,
        inputs=[TensorPort("latents")], outputs=[TensorPort("frames")],
        state_reads=["vae_cache"], state_writes=["vae_cache", "video_out"],
        stream_mode=StreamMode.STREAMING,
    )
