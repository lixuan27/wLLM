"""IR operators for the LongLive pipeline.

Two granularities are defined here:

* **Model-level ops** — the fine-grained per-chunk compute inside the exposed
  DiT + VAE stages (noise sample, DMD denoise steps, clean-K/V cache write,
  per-frame causal VAE decode) plus the prompt-scoped preamble (text encode,
  DiT cross-attn prefill). These are what the Phase-2 executor validates and
  what the analysis tools inspect for pipeline / streaming / cross-chunk
  structure.

* **Worker-level ops** — the coarse pipeline stages (VAD segment, black-box
  ASR, the exposed video-gen composite). These document stage scheduling.

Every op delegates its numerical work to the shared ``LongLiveCore``
(``engine.py``), so the IR is the *same* computation as the reference and the
deployment variants. State is accessed through the executor's ``StateStore``
purely to exercise the declared-dependency contract; the actual tensors live
inside the runners (``kv_cache`` mutated by ``DiTRunner.run``,
``vae_feat_cache`` mutated by ``VAERunner.run``), which is the in-place-mutation
pattern the IR framework documents.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from wllm.serving.ir import ir_operator
from wllm.serving.ir.graph import IROperator, OpType, StreamMode, TensorPort

from wllm.apps.longlive.backend.rocm.ir.engine import ChunkIndices, LongLiveCore


# State object names (see graph_builder for their StateObject declarations)
S_KV_CACHE = "kv_cache"             # DiT sliding-window ring (chunk_persistent)
S_VAE_FEAT = "vae_feat_cache"       # VAE causal feature cache (chunk_persistent)
S_NOISE_GEN = "noise_gen"           # torch.Generator RNG position (chunk_persistent)
S_ENCODER_KV = "encoder_kv"         # DiT cross-attn K/V from prompt (session_init)


class LongLiveIRContext:
    """Opaque context handed to every op's ``execute``.

    Owns the ``LongLiveCore`` (runners + counters) and the per-chunk index
    bundle. A scheduler (the executor harness, or a deployment worker) calls
    ``begin_chunk()`` before ``run_chunk`` and ``end_chunk()`` after, so the
    ring-buffer / RoPE arithmetic stays outside the state-dependency graph
    (it is a deterministic function of the session counters, mirroring how a
    real deployment computes slot indices on its scheduler)."""

    def __init__(self, core: LongLiveCore) -> None:
        self.core = core
        self.idx: Optional[ChunkIndices] = None
        # worker-graph helpers (optional)
        self.asr_model: Any = None
        self.segmenter: Any = None
        self.sample_rate: int = int(core.cfg.audio_sample_rate)

    def begin_chunk(self) -> None:
        self.idx = self.core.compute_chunk_indices()

    def end_chunk(self) -> None:
        assert self.idx is not None
        self.core.advance_chunk(self.idx)
        self.idx = None


# ----------------------------------------------------------------------------
# Model-level chunk operators (subclassed: parameterized by index)
# ----------------------------------------------------------------------------
class SampleNoiseOp(IROperator):
    def __init__(self) -> None:
        super().__init__(
            name="sample_noise",
            op_type=OpType.EXPOSED,
            inputs=[],
            outputs=[TensorPort("latents")],
            state_reads=[],
            state_writes=[S_NOISE_GEN],
            stream_mode=StreamMode.BATCH,
        )

    def execute(self, inputs, context, state):
        state.get(S_NOISE_GEN)  # acknowledge RNG advance
        return {"latents": context.core.sample_noise()}


class DiTDenoiseStep(IROperator):
    """One DMD denoise step. Steps that re-noise (all but the last) advance the
    RNG, so they additionally write ``noise_gen``."""

    def __init__(self, step_idx: int, num_steps: int) -> None:
        self.step_idx = step_idx
        draws_noise = step_idx < num_steps - 1
        writes = [S_KV_CACHE] + ([S_NOISE_GEN] if draws_noise else [])
        super().__init__(
            name=f"denoise_step_{step_idx}",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("latents_in")],
            outputs=[TensorPort("latents_out")],
            state_reads=[S_ENCODER_KV],
            state_writes=writes,
            stream_mode=StreamMode.BATCH,
        )

    def execute(self, inputs, context, state):
        state.get(S_KV_CACHE)  # DiT writes this chunk's K/V into the ring
        latents = context.core.denoise_step(
            inputs["latents_in"], self.step_idx, context.idx
        )
        return {"latents_out": latents}


class DiTCacheWrite(IROperator):
    """t=0 pass writing this chunk's *clean* K/V into the ring."""

    def __init__(self) -> None:
        super().__init__(
            name="cache_write",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("latents_in")],
            outputs=[],
            state_reads=[S_ENCODER_KV],
            state_writes=[S_KV_CACHE],
            stream_mode=StreamMode.BATCH,
        )

    def execute(self, inputs, context, state):
        state.get(S_KV_CACHE)
        context.core.cache_write(inputs["latents_in"], context.idx)
        return {}


class VAEDecodeFrame(IROperator):
    """Causal VAE decode of one latent frame. Chained through the shared
    ``vae_feat_cache`` (ordering edges enforce the 0..chunk-1 sequence)."""

    def __init__(self, local_frame: int) -> None:
        self.local_frame = local_frame
        super().__init__(
            name=f"vae_decode_{local_frame}",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("latents_in")],
            outputs=[TensorPort(f"frame_{local_frame}")],
            state_reads=[],
            state_writes=[S_VAE_FEAT],
            stream_mode=StreamMode.STREAMING,
        )

    def execute(self, inputs, context, state):
        state.get(S_VAE_FEAT)
        pixels = context.core.decode_frame(inputs["latents_in"], self.local_frame)
        return {f"frame_{self.local_frame}": pixels}


# ----------------------------------------------------------------------------
# Preamble ops (prompt-scoped): text encode + DiT cross-attn prefill
# ----------------------------------------------------------------------------
@ir_operator(
    name="text_encode",
    op_type=OpType.EXPOSED,
    inputs=[TensorPort("prompt")],
    outputs=[TensorPort("prompt_embeds")],
    state_reads=[],
    state_writes=[],
)
def text_encode(inputs, context, state):
    embeds = context.core._get_t5_prompt_embeds(inputs["prompt"])
    return {"prompt_embeds": embeds}


@ir_operator(
    name="dit_encode",
    op_type=OpType.EXPOSED,
    inputs=[TensorPort("prompt_embeds")],
    outputs=[],
    state_reads=[],
    state_writes=[S_ENCODER_KV],
)
def dit_encode(inputs, context, state):
    # dit_encode (re)fills the cross-attention KV from the new prompt -> a
    # genuine rebind of the encoder_kv state, so use set (not in-place get).
    state.set(S_ENCODER_KV, True)
    context.core._prompt_embeds = inputs["prompt_embeds"]
    if context.core.dit_runner is not None:
        context.core.dit_runner.encode(inputs["prompt_embeds"])
    return {}


# ----------------------------------------------------------------------------
# Worker-level ops (coarse stage graph)
# ----------------------------------------------------------------------------
@ir_operator(
    name="vad_segment",
    op_type=OpType.EXPOSED,
    inputs=[TensorPort("audio_chunks")],
    outputs=[TensorPort("utterance_audio")],
    state_writes=["vad_state"],
)
def vad_segment(inputs, context, state):
    """Run the streaming VAD over a list of 320-sample audio chunks; emit the
    concatenated utterance audio once an utterance completes (else None)."""
    state.get("vad_state")
    seg = context.segmenter
    utterance = None
    for chunk in inputs["audio_chunks"]:
        should_infer, audio = seg.process_chunk(chunk)
        if should_infer and audio is not None:
            utterance = audio
    return {"utterance_audio": utterance}


@ir_operator(
    name="asr_transcribe",
    op_type=OpType.BLACK_BOX,
    inputs=[TensorPort("utterance_audio")],
    outputs=[TensorPort("prompt")],
)
def asr_transcribe(inputs, context, state):
    audio = inputs["utterance_audio"]
    if audio is None:
        return {"prompt": None}
    results = context.asr_model.transcribe(
        audio=(np.asarray(audio, dtype=np.float32).reshape(-1), context.sample_rate),
        language="English",
    )
    text = (results[0].text or "").strip()
    return {"prompt": text or None}
