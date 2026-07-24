"""Build LongLive IR graphs.

``build_model_graph`` produces the executable model-level graph (one
``step()`` decomposed into noise/denoise/cache/VAE ops) that Phase-2
validation runs through the SequentialExecutor.

``build_worker_graph`` produces the high-level pipeline-scheduling graph
(audio/VAD → ASR black box → prompt encode → DiT chunk → VAE chunk).
"""
from __future__ import annotations

from wllm.serving.ir import (IRGraph, IREdge, StateObject, TensorPort,
                        StreamingInfo, StreamingPattern)
from wllm.serving.rt_config import RTConfig

from wllm.apps.longlive.backend.cuda.ir import ops as O


def build_model_graph(cfg: RTConfig) -> IRGraph:
    g = IRGraph(name="longlive_model")

    g.add_state(StateObject("kv_ring", "DiT sliding-window K/V ring (per-layer)",
                            scope="chunk_persistent"))
    g.add_state(StateObject("ring_state", "LongLive ring bookkeeping "
                            "(block_idx, rolling_writes, slots, rope offset)",
                            scope="chunk_persistent"))
    g.add_state(StateObject("encoder_kv", "Cross-attention KV from the prompt",
                            scope="session_init"))
    g.add_state(StateObject("vae_cache", "Wan VAE causal feat cache + decode count",
                            scope="chunk_persistent"))
    g.add_state(StateObject("video_out", "Emitted video frames sink",
                            scope="chunk_persistent"))

    # preamble
    g.add_operator(O.EncodePrompt())
    g.preamble_ops = ["encode_prompt"]
    g.external_inputs = [TensorPort("prompt")]

    # chunk ops
    g.add_operator(O.ChunkPlan())
    g.add_operator(O.NoiseSample())
    num_steps = int(cfg.num_inference_steps)
    for k in range(num_steps):
        g.add_operator(O.DenoiseStep(k))
    g.add_operator(O.CacheWrite())
    chunk_size = int(cfg.chunk_size)
    for l in range(chunk_size):
        g.add_operator(O.VaeDecode(l))

    chunk = ["chunk_plan", "noise_sample"]
    chunk += [f"denoise_{k}" for k in range(num_steps)]
    chunk += ["cache_write"]
    chunk += [f"vae_decode_{l}" for l in range(chunk_size)]
    g.chunk_ops = chunk

    # edges: plan -> every denoise + cache
    for k in range(num_steps):
        g.add_edge(IREdge("chunk_plan", "plan", f"denoise_{k}", "plan"))
    g.add_edge(IREdge("chunk_plan", "plan", "cache_write", "plan"))

    # latents chain
    g.add_edge(IREdge("noise_sample", "latents", "denoise_0", "latents"))
    for k in range(num_steps - 1):
        g.add_edge(IREdge(f"denoise_{k}", "latents", f"denoise_{k+1}", "latents"))
    g.add_edge(IREdge(f"denoise_{num_steps-1}", "latents", "cache_write", "latents"))

    # cache_write clean latents -> each VAE decode (streaming: 1 chunk -> 8 frames)
    stream = StreamingInfo(
        pattern=StreamingPattern.FIXED_RATE, src_chunk_size=1, dst_chunk_size=1,
        description="clean latents stream one frame at a time into the VAE decoder",
    )
    for l in range(chunk_size):
        g.add_edge(IREdge("cache_write", "latents", f"vae_decode_{l}", "latents",
                          streaming=stream if l == 0 else None))

    errs = g.validate()
    if errs:
        raise ValueError("model graph invalid:\n  - " + "\n  - ".join(errs))
    return g


def build_worker_graph(cfg: RTConfig) -> IRGraph:
    g = IRGraph(name="longlive_worker")
    g.add_state(StateObject("vad_state", "Streaming VAD segmenter state",
                            scope="chunk_persistent"))
    g.add_state(StateObject("encoder_kv", "Cross-attention KV from the prompt",
                            scope="session_init"))
    g.add_state(StateObject("kv_ring", "DiT sliding-window K/V ring",
                            scope="chunk_persistent"))
    g.add_state(StateObject("ring_state", "LongLive ring bookkeeping",
                            scope="chunk_persistent"))
    g.add_state(StateObject("vae_cache", "Wan VAE causal feat cache + count",
                            scope="chunk_persistent"))
    g.add_state(StateObject("video_out", "Emitted video frames sink",
                            scope="chunk_persistent"))

    for op in (O.audio_vad_op(), O.asr_op(), O.apply_prompt_op(),
               O.dit_chunk_op(), O.vae_chunk_op()):
        g.add_operator(op)

    g.external_inputs = [TensorPort("audio_chunk")]
    g.chunk_ops = ["audio_vad", "asr", "apply_prompt", "dit_chunk", "vae_chunk"]

    # audio is variable-rate streaming into ASR (utterance boundaries are
    # content-dependent — VAD decides when an utterance ends).
    g.add_edge(IREdge("audio_vad", "utterance", "asr", "utterance",
                      streaming=StreamingInfo(
                          pattern=StreamingPattern.VARIABLE_RATE,
                          description="VAD emits an utterance only at speech end")))
    g.add_edge(IREdge("asr", "prompt_text", "apply_prompt", "prompt_text"))
    g.add_edge(IREdge("apply_prompt", "encoder_kv", "dit_chunk", "encoder_kv"))
    # DiT streams clean latents into VAE (1 latent chunk -> chunk_size frames).
    g.add_edge(IREdge("dit_chunk", "latents", "vae_chunk", "latents",
                      streaming=StreamingInfo(
                          pattern=StreamingPattern.FIXED_RATE,
                          src_chunk_size=int(cfg.chunk_size),
                          dst_chunk_size=int(cfg.chunk_size),
                          description="DiT emits chunk_size clean latents; VAE "
                                      "decodes them frame-by-frame (4 px frames "
                                      "per latent, causal)")))
    g.sub_graphs["longlive_model"] = build_model_graph(cfg)

    errs = g.validate()
    if errs:
        raise ValueError("worker graph invalid:\n  - " + "\n  - ".join(errs))
    return g
