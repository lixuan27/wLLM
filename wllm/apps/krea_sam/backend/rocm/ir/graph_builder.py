"""Build IR graphs for the Krea-Realtime + SAM3 pipeline.

`build_worker_graph` is the full exposed computation graph for one chunk: the
worker-level stage scheduling *and* the fine-grained DiT model-level ops
(cache-fill + per-step denoise) and VAE ops. `build_dit_model_graph` /
`build_vae_model_graph` are focused sub-views used for documentation of the
within-model structure.

State-object scopes encode the true cross-chunk dependency structure the
analysis relies on (see ir/ops.py docstring).
"""

from __future__ import annotations

from wllm.serving.ir import IREdge, IRGraph, StateObject, StreamingInfo, StreamingPattern, TensorPort

from wllm.apps.krea_sam.backend.rocm.ir import ops as O


def _add_states(g: IRGraph):
    g.add_state(StateObject("clean_latent_context", "rolling denoised-latent context conditioning the DiT", "chunk_persistent"))
    g.add_state(StateObject("dit_kv_cache", "DiT self-attn KV cache (refilled from clean_latent_context each chunk)", "chunk_persistent"))
    g.add_state(StateObject("encoder_kv_cache", "text cross-attn KV, filled once at session init", "session_init"))
    g.add_state(StateObject("vae_encoder_cache", "streaming causal VAE encoder temporal cache", "chunk_persistent"))
    g.add_state(StateObject("vae_decoder_cache", "causal VAE decoder temporal cache", "chunk_persistent"))
    g.add_state(StateObject("sam_tracker_state", "SAM3 per-session tracking memory", "chunk_persistent"))


def build_worker_graph(cfg) -> IRGraph:
    n_steps = int(cfg.num_inference_steps)
    chunk = int(cfg.chunk_size)
    scale_t = int(cfg.vae_config.scale_factor_temporal)

    g = IRGraph(name="krea_sam_worker")
    _add_states(g)

    g.external_inputs = [TensorPort("input_pixels", ("T", "C", "H", "W")),
                         TensorPort("raw_frames", ("T", "H", "W", 3))]
    g.external_outputs = [TensorPort("composited", ("T", "H", "W", 3))]

    # ---- preamble ----
    g.add_operator(O.session_init)
    g.preamble_ops = ["session_init"]

    # ---- chunk ops ----
    g.add_operator(O.VaeEncode())
    g.add_operator(O.prepare_noisy)
    g.add_operator(O.DitCacheFill())
    denoise_names = []
    for k in range(n_steps):
        g.add_operator(O.DitDenoiseStep(k, n_steps))
        denoise_names.append(f"dit_denoise_{k}")
    g.add_operator(O.DitAppendContext())
    g.add_operator(O.VaeDecode())
    g.add_operator(O.SamSegment())
    g.add_operator(O.Composite())

    g.chunk_ops = (["vae_encode", "prepare_noisy", "dit_cache_fill"] + denoise_names
                   + ["dit_append_context", "vae_decode", "sam_segment", "composite"])

    # ---- data edges: Krea spine ----
    # external input_pixels feeds vae_encode (resolved by name); raw_frames feeds sam/composite by name.
    g.add_edge(IREdge("vae_encode", "input_latents", "prepare_noisy", "input_latents",
                      streaming=StreamingInfo(StreamingPattern.FIXED_RATE, scale_t, 1,
                                              "streaming causal encode: scale_t pixel frames -> 1 latent frame")))
    g.add_edge(IREdge("prepare_noisy", "noisy_latents", denoise_names[0], "latents_in"))
    for k in range(n_steps):
        g.add_edge(IREdge("dit_cache_fill", "context_tokens", denoise_names[k], "context_tokens"))
    for k in range(n_steps - 1):
        g.add_edge(IREdge(denoise_names[k], "latents_out", denoise_names[k + 1], "latents_in"))
    last = denoise_names[-1]
    g.add_edge(IREdge(last, "denoised", "dit_append_context", "denoised"))
    g.add_edge(IREdge(last, "denoised", "vae_decode", "denoised"))

    # vae_decode -> composite: streaming (frames emitted incrementally)
    g.add_edge(IREdge("vae_decode", "krea_frames", "composite", "krea_frames",
                      streaming=StreamingInfo(StreamingPattern.FIXED_RATE, 1, 1,
                                              "per-frame causal decode streams frames to composite")))
    # raw_frames -> sam_segment, composite  (external inputs by name)
    g.add_edge(IREdge("sam_segment", "masks", "composite", "masks"))

    # ordering: cache_fill must precede denoise_0 (captured by context_tokens data edge already);
    # composite is the sink producing the external output.

    errors = g.validate()
    if errors:
        raise ValueError("worker graph invalid:\n  " + "\n  ".join(errors))
    return g


def build_dit_model_graph(cfg) -> IRGraph:
    """Focused model-level view of the exposed DiT: cache-fill + per-step
    denoise. Surfaces that the denoise steps form a serial chain sharing the
    dit_kv_cache (chunk-persistent recurrence via clean_latent_context)."""
    n_steps = int(cfg.num_inference_steps)
    g = IRGraph(name="krea_dit_model")
    g.add_state(StateObject("clean_latent_context", "clean-context recurrence", "chunk_persistent"))
    g.add_state(StateObject("dit_kv_cache", "DiT KV cache", "chunk_persistent"))
    g.add_state(StateObject("encoder_kv_cache", "text cross-attn KV", "session_init"))
    g.external_inputs = [TensorPort("noisy_latents")]
    g.external_outputs = [TensorPort("denoised")]

    g.add_operator(O.DitCacheFill())
    names = []
    for k in range(n_steps):
        g.add_operator(O.DitDenoiseStep(k, n_steps))
        names.append(f"dit_denoise_{k}")
    g.add_operator(O.DitAppendContext())
    g.chunk_ops = ["dit_cache_fill"] + names + ["dit_append_context"]

    g.add_edge(IREdge("dit_cache_fill", "context_tokens", names[0], "context_tokens"))
    for k in range(1, n_steps):
        g.add_edge(IREdge("dit_cache_fill", "context_tokens", names[k], "context_tokens"))
    for k in range(n_steps - 1):
        g.add_edge(IREdge(names[k], "latents_out", names[k + 1], "latents_in"))
    g.add_edge(IREdge(names[-1], "denoised", "dit_append_context", "denoised"))
    # noisy_latents external input feeds denoise_0.latents_in by name
    errors = g.validate()
    if errors:
        raise ValueError("dit model graph invalid:\n  " + "\n  ".join(errors))
    return g


def build_vae_model_graph(cfg) -> IRGraph:
    """Focused model-level view of the exposed VAE: encode (streaming) and
    decode (causal per-frame). Surfaces that encode and decode use *disjoint*
    caches (independent -> can overlap across chunks)."""
    g = IRGraph(name="krea_vae_model")
    g.add_state(StateObject("vae_encoder_cache", "streaming causal encoder cache", "chunk_persistent"))
    g.add_state(StateObject("vae_decoder_cache", "causal decoder cache", "chunk_persistent"))
    g.external_inputs = [TensorPort("input_pixels"), TensorPort("denoised")]
    g.external_outputs = [TensorPort("krea_frames")]
    g.add_operator(O.VaeEncode())
    g.add_operator(O.VaeDecode())
    g.chunk_ops = ["vae_encode", "vae_decode"]
    # no edge between them: they are independent (disjoint caches); decode
    # consumes `denoised` (from the DiT), encode produces `input_latents`.
    errors = g.validate()
    if errors:
        raise ValueError("vae model graph invalid:\n  " + "\n  ".join(errors))
    return g
