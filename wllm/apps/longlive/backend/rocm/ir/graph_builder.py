"""Build the LongLive IR graphs.

``build_model_graph(cfg)`` — the fine-grained per-chunk DiT + VAE compute graph
(the one the Phase-2 executor validates and the analysis tools inspect).

``build_worker_graph(cfg)`` — the coarse worker stage graph (audio -> VAD ->
black-box ASR -> exposed video-gen composite), documenting stage scheduling.
"""

from __future__ import annotations

from wllm.serving.ir.graph import (
    IRGraph,
    IREdge,
    OpType,
    StateObject,
    StreamingInfo,
    StreamingPattern,
    StreamMode,
    TensorPort,
)
from wllm.serving.rt_config import RTConfig

from wllm.apps.longlive.backend.rocm.ir import ops as O


def build_model_graph(cfg: RTConfig) -> IRGraph:
    num_steps = int(cfg.num_inference_steps)
    chunk_size = int(cfg.chunk_size)
    kv_spatial = int(cfg.kv_spatial)

    g = IRGraph(name="longlive_model")

    # ---- state objects ----
    g.add_state(StateObject(
        S := O.S_KV_CACHE,
        description="DiT sliding-window KV ring (sink + rolling window); "
                    "every DiT pass writes the chunk's K/V at its slot and "
                    "attends [0:cache_end]; the t=0 pass persists the clean K/V.",
        scope="chunk_persistent",
    ))
    g.add_state(StateObject(
        O.S_VAE_FEAT,
        description="Causal VAE feature cache; makes per-frame decode a "
                    "sequential stream within and across chunks.",
        scope="chunk_persistent",
    ))
    g.add_state(StateObject(
        O.S_NOISE_GEN,
        description="Seeded torch.Generator RNG position; draws must stay in "
                    "reference order (initial noise then per-step re-noise).",
        scope="chunk_persistent",
    ))
    g.add_state(StateObject(
        O.S_ENCODER_KV,
        description="DiT cross-attention K/V from the current prompt; refilled "
                    "by dit_encode, read by every DiT pass.",
        scope="session_init",
    ))

    # ---- preamble ops (prompt-scoped) ----
    g.add_operator(O.text_encode)
    g.add_operator(O.dit_encode)
    g.preamble_ops = ["text_encode", "dit_encode"]
    g.add_edge(IREdge("text_encode", "prompt_embeds", "dit_encode", "prompt_embeds"))
    g.external_inputs.append(TensorPort("prompt"))

    # ---- chunk ops ----
    g.add_operator(O.SampleNoiseOp())
    for k in range(num_steps):
        g.add_operator(O.DiTDenoiseStep(k, num_steps))
    g.add_operator(O.DiTCacheWrite())
    for l in range(chunk_size):
        g.add_operator(O.VAEDecodeFrame(l))

    chunk_ops = (
        ["sample_noise"]
        + [f"denoise_step_{k}" for k in range(num_steps)]
        + ["cache_write"]
        + [f"vae_decode_{l}" for l in range(chunk_size)]
    )
    g.chunk_ops = chunk_ops

    # ---- data edges: noise -> denoise chain ----
    g.add_edge(IREdge("sample_noise", "latents", "denoise_step_0", "latents_in"))
    for k in range(num_steps - 1):
        g.add_edge(IREdge(
            f"denoise_step_{k}", "latents_out",
            f"denoise_step_{k + 1}", "latents_in",
        ))
    last_denoise = f"denoise_step_{num_steps - 1}"

    # clean latents feed both the cache-write and every VAE frame decode.
    # The DiT produces all `chunk_size` clean latents as a batch, then the VAE
    # consumes them one frame at a time -> fixed-rate streaming handoff.
    g.add_edge(IREdge(last_denoise, "latents_out", "cache_write", "latents_in"))
    for l in range(chunk_size):
        g.add_edge(IREdge(
            last_denoise, "latents_out", f"vae_decode_{l}", "latents_in",
            streaming=(StreamingInfo(
                pattern=StreamingPattern.FIXED_RATE,
                src_chunk_size=chunk_size,
                dst_chunk_size=1,
                description="DiT emits a chunk of clean latents; VAE decodes "
                            "1 latent frame at a time (causal).",
            ) if l == 0 else None),
        ))

    # ordering edges: VAE decode is a strict causal sequence (shared feat cache)
    for l in range(chunk_size - 1):
        g.add_edge(IREdge(f"vae_decode_{l}", None, f"vae_decode_{l + 1}", None))

    # ---- external outputs: one pixel-frame port per latent frame ----
    for l in range(chunk_size):
        g.external_outputs.append(TensorPort(f"frame_{l}"))

    errors = g.validate()
    if errors:
        raise ValueError("model graph invalid:\n  - " + "\n  - ".join(errors))
    _ = (S, kv_spatial)  # silence linters; documented above
    return g


def build_worker_graph(cfg: RTConfig) -> IRGraph:
    """Coarse worker stage graph. ``video_gen`` is a COMPOSITE stage whose
    sub-graph is the model graph above."""
    g = IRGraph(name="longlive_worker")

    g.add_state(StateObject(
        "vad_state",
        description="Streaming VAD accumulator (speech/silence buffers, "
                    "utterance frames).",
        scope="chunk_persistent",
    ))
    g.add_state(StateObject(
        "video_stream",
        description="The generated RGB frame stream written to the shm video "
                    "buffer (append-only across chunks).",
        scope="chunk_persistent",
    ))

    g.add_operator(O.vad_segment)
    g.add_operator(O.asr_transcribe)

    # exposed video-gen composite stage (sub-graph = model graph)
    from wllm.serving.ir.graph import IROperator

    class VideoGenStage(IROperator):
        def __init__(self):
            super().__init__(
                name="video_gen",
                op_type=OpType.COMPOSITE,
                inputs=[TensorPort("prompt")],
                outputs=[TensorPort("frames")],
                state_reads=[],
                state_writes=["video_stream"],
                stream_mode=StreamMode.STREAMING,
                sub_graph="video_gen",
            )

        def execute(self, inputs, context, state):  # pragma: no cover - documentation stage
            raise NotImplementedError(
                "worker graph is a scheduling document; validate via the model "
                "graph"
            )

    g.add_operator(VideoGenStage())
    g.sub_graphs["video_gen"] = build_model_graph(cfg)

    g.chunk_ops = ["vad_segment", "asr_transcribe", "video_gen"]
    g.external_inputs.append(TensorPort("audio_chunks"))
    g.external_outputs.append(TensorPort("frames"))

    # audio -> VAD -> (utterance) -> ASR -> (prompt) -> video-gen.
    # The audio->VAD edge is a variable-rate stream: 320-sample chunks arrive
    # continuously; an utterance boundary is content-dependent (VAD-detected).
    g.add_edge(IREdge(
        "vad_segment", "utterance_audio", "asr_transcribe", "utterance_audio",
        streaming=StreamingInfo(
            pattern=StreamingPattern.VARIABLE_RATE,
            description="VAD emits an utterance only when trailing silence "
                        "closes it (content-dependent boundary).",
        ),
    ))
    g.add_edge(IREdge("asr_transcribe", "prompt", "video_gen", "prompt"))

    errors = g.validate()
    if errors:
        raise ValueError("worker graph invalid:\n  - " + "\n  - ".join(errors))
    return g
