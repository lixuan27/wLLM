"""IR graph builders for the Krea+SAM pipeline.

``build_krea_model_graph`` — the fine-grained model-level graph for the
Krea v2v stage (the exposed DiT + VAE computation).

``build_worker_graph`` — the high-level worker graph (krea_v2v ‖
sam_segment → composite). ``krea_v2v`` is a COMPOSITE node whose
sub-graph is the Krea model graph; at execution time it delegates to a
nested ``SequentialExecutor`` held on the context.

Both are chunk-periodic. The state-object scopes are what the analysis
tools read to derive pipeline stages and cross-chunk dependencies.
"""

from __future__ import annotations

from wllm.serving.ir import (
    IREdge,
    IRGraph,
    StateObject,
    StreamingInfo,
    StreamingPattern,
    TensorPort,
)

from wllm.apps.krea_sam.backend.cuda.ir import ops as K


def build_krea_model_graph(num_inference_steps: int) -> IRGraph:
    g = IRGraph(name="krea_model")

    # Cross-chunk persistent state (each is a distinct causal cache /
    # context buffer). Distinct names => the analysis can place the three
    # caches on separate devices and pipeline them across chunks.
    g.add_state(StateObject("clean_latent_context", "denoised-latent context feeding the DiT prefix", "chunk_persistent"))
    g.add_state(StateObject("dit_kv", "DiT prefix+chunk KV cache (in dit_runner)", "chunk_persistent"))
    g.add_state(StateObject("vae_enc_cache", "streaming causal encoder temporal cache", "chunk_persistent"))
    g.add_state(StateObject("vae_dec_cache", "causal decoder temporal cache", "chunk_persistent"))

    g.add_operator(K.vae_encode)
    g.add_operator(K.add_noise)
    g.add_operator(K.fill_context)
    steps = [K.DiTDenoiseStep(i, num_inference_steps) for i in range(num_inference_steps)]
    for s in steps:
        g.add_operator(s)
    g.add_operator(K.append_context)
    g.add_operator(K.vae_decode)

    g.preamble_ops = []
    g.chunk_ops = (
        ["vae_encode", "add_noise", "fill_context"]
        + [s.name for s in steps]
        + ["append_context", "vae_decode"]
    )

    g.external_inputs = [TensorPort("input_frames")]
    g.external_outputs = [TensorPort("krea_frames")]

    # data edges
    g.add_edge(IREdge("vae_encode", "input_latents", "add_noise", "input_latents",
                      streaming=StreamingInfo(StreamingPattern.FIXED_RATE,
                                              src_chunk_size=1, dst_chunk_size=1,
                                              description="streaming causal VAE encode")))
    g.add_edge(IREdge("vae_encode", "input_latents", "fill_context", "input_latents"))
    g.add_edge(IREdge("add_noise", "noisy_latents", steps[0].name, "latents_in"))
    for i, s in enumerate(steps):
        g.add_edge(IREdge("fill_context", "context_tokens", s.name, "context_tokens"))
        if i + 1 < len(steps):
            g.add_edge(IREdge(s.name, "latents_out", steps[i + 1].name, "latents_in"))
    last = steps[-1].name
    g.add_edge(IREdge(last, "latents_out", "append_context", "denoised_latents"))
    g.add_edge(IREdge(last, "latents_out", "vae_decode", "denoised_latents",
                      streaming=StreamingInfo(StreamingPattern.FIXED_RATE,
                                              src_chunk_size=1, dst_chunk_size=1,
                                              description="per-latent-frame causal decode (streamable)")))
    return g


def build_worker_graph(num_inference_steps: int) -> IRGraph:
    g = IRGraph(name="krea_sam_worker")

    g.add_state(StateObject("clean_latent_context", "Krea denoised-latent context", "chunk_persistent"))
    g.add_state(StateObject("dit_kv", "DiT KV cache", "chunk_persistent"))
    g.add_state(StateObject("vae_enc_cache", "VAE encoder cache", "chunk_persistent"))
    g.add_state(StateObject("vae_dec_cache", "VAE decoder cache", "chunk_persistent"))
    g.add_state(StateObject("sam_tracking", "SAM per-session tracking memory", "chunk_persistent"))

    g.add_operator(K.krea_v2v)
    g.add_operator(K.sam_segment)
    g.add_operator(K.composite)

    g.preamble_ops = []
    g.chunk_ops = ["krea_v2v", "sam_segment", "composite"]

    g.external_inputs = [TensorPort("input_frames"), TensorPort("raw_frames")]
    g.external_outputs = [TensorPort("composited")]

    # krea_v2v ‖ sam_segment (no shared state) -> composite
    g.add_edge(IREdge("krea_v2v", "krea_frames", "composite", "krea_frames",
                      streaming=StreamingInfo(StreamingPattern.VARIABLE_RATE,
                                              description="krea frames stream to compositor")))
    g.add_edge(IREdge("sam_segment", "masks", "composite", "masks",
                      streaming=StreamingInfo(StreamingPattern.VARIABLE_RATE,
                                              description="sam masks stream to compositor")))

    g.sub_graphs = {"krea_model": build_krea_model_graph(num_inference_steps)}
    return g
