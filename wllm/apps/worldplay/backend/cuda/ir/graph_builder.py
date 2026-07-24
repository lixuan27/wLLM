"""Build IR graphs for the WorldPlay reactive-decoding pipeline.

Two graphs, per repo AGENTS.md Phase 1:

* ``build_model_graph(cfg)`` -- the fine-grained, per-chunk *model-level* graph
  (camera ingest -> KV-fill -> denoise x N -> write-back -> VAE decode x C ->
  collect). This is the graph the SequentialExecutor runs for Phase-2
  validation and the graph the analysis tools chew on. Its state structure
  (``kv``/``latents`` for the DiT vs ``vae`` for the decoder) is what lets
  ``find_pipeline_stages`` recognise the DiT and VAE as separable, pipelinable
  stages.

* ``build_worker_graph(cfg)`` -- the coarse *worker-level* graph
  (ingest -> dit_denoise[COMPOSITE] -> vae_decode[COMPOSITE] -> video_write),
  with the DiT and VAE stages descending into ``dit_model`` / ``vae_model``
  sub-graphs. This is the high-level stage-scheduling view, with the streaming
  edge from the DiT (BATCH: all latents materialise together) to the VAE
  (STREAMING: frames trickle out latent-by-latent) made explicit.
"""

from __future__ import annotations

from wllm.serving.ir import (
    IRGraph, IREdge, StateObject, TensorPort, StreamingInfo, StreamingPattern,
)

from wllm.apps.worldplay.backend.cuda.ir.ops import (
    IngestActions, KVFill, DenoiseStep, WriteBackLatents,
    VAEDecodeLatent, CollectFrames, DiTStage, VAEStage, VideoWrite,
)


_STATE = [
    ("cam", "camera accumulators (viewmats/Ks/action + running pose T/C_inv)", "chunk_persistent"),
    ("latents", "latent history buffer (_latents)", "chunk_persistent"),
    ("kv", "DiT rope/prope self-attention KV cache", "chunk_persistent"),
    ("vae", "VAE decoder causal feature cache", "chunk_persistent"),
    ("enc_kv", "cross-attention encoder KV (text), filled once at session init", "session_init"),
    ("first_img", "VAE-encoded first-image condition, set once at session init", "session_init"),
]


def _add_state(g: IRGraph, names):
    for n, desc, scope in _STATE:
        if n in names:
            g.add_state(StateObject(name=n, description=desc, scope=scope))


def build_model_graph(cfg) -> IRGraph:
    """Fine-grained per-chunk graph (validated + analysed)."""
    g = IRGraph(name="worldplay_chunk")
    _add_state(g, {"cam", "latents", "kv", "vae", "enc_kv", "first_img"})

    ns = int(cfg.num_inference_steps)
    cs = int(cfg.chunk_size)

    ingest = IngestActions()
    kv_fill = KVFill()
    steps = [DenoiseStep(i) for i in range(ns)]
    wb = WriteBackLatents()
    decs = [VAEDecodeLatent(j, cs) for j in range(cs)]
    collect = CollectFrames(cs)

    ordered = [ingest, kv_fill, *steps, wb, *decs, collect]
    for op in ordered:
        g.add_operator(op)

    g.external_inputs = [TensorPort("action_codes")]
    g.external_outputs = [TensorPort("chunk_video")]
    g.chunk_ops = [op.name for op in ordered]

    # ingest feeds both the (conditional) KV-fill and the first denoise step,
    # so denoise_0 always has a resolvable data input even on chunk 0 when
    # kv_fill is skipped.
    g.add_edge(IREdge("ingest_actions", "ready", "kv_fill", "ready"))
    g.add_edge(IREdge("ingest_actions", "ready", "denoise_step_0", "prev"))
    g.add_edge(IREdge("kv_fill", None, "denoise_step_0", None))  # ordering only

    for i in range(1, ns):
        g.add_edge(IREdge(f"denoise_step_{i-1}", "stepped", f"denoise_step_{i}", "prev"))
    g.add_edge(IREdge(f"denoise_step_{ns-1}", "stepped", "writeback_latents", "stepped"))

    stream = lambda: StreamingInfo(
        pattern=StreamingPattern.FIXED_RATE, src_chunk_size=cs, dst_chunk_size=1,
        description=("DiT writes all chunk_size latents together (BATCH); the VAE "
                     "consumes one latent at a time and emits ~scale_factor_temporal "
                     "frames per latent, so frames can stream to the video buffer "
                     "as each latent decodes instead of waiting for the whole chunk."),
    )
    for j in range(cs):
        g.add_edge(IREdge("writeback_latents", "chunk_latents", f"vae_decode_{j}",
                          "chunk_latents", streaming=stream()))
    for j in range(1, cs):  # causal feat-cache chains the decodes
        g.add_edge(IREdge(f"vae_decode_{j-1}", None, f"vae_decode_{j}", None))
    for j in range(cs):
        g.add_edge(IREdge(f"vae_decode_{j}", "frames", "collect_frames", f"f{j}"))

    errs = g.validate()
    if errs:
        raise ValueError("model graph invalid:\n  - " + "\n  - ".join(errs))
    return g


def _build_dit_subgraph(cfg) -> IRGraph:
    g = IRGraph(name="dit_model")
    _add_state(g, {"cam", "latents", "kv", "enc_kv"})
    ns = int(cfg.num_inference_steps)
    kv_fill = KVFill()
    steps = [DenoiseStep(i) for i in range(ns)]
    wb = WriteBackLatents()
    ordered = [kv_fill, *steps, wb]
    for op in ordered:
        g.add_operator(op)
    g.external_inputs = [TensorPort("ready")]
    g.external_outputs = [TensorPort("chunk_latents")]
    g.chunk_ops = [op.name for op in ordered]
    g.add_edge(IREdge("kv_fill", None, "denoise_step_0", None))
    # external 'ready' feeds denoise_0 directly
    for i in range(1, ns):
        g.add_edge(IREdge(f"denoise_step_{i-1}", "stepped", f"denoise_step_{i}", "prev"))
    g.add_edge(IREdge(f"denoise_step_{ns-1}", "stepped", "writeback_latents", "stepped"))
    return g


def _build_vae_subgraph(cfg) -> IRGraph:
    g = IRGraph(name="vae_model")
    _add_state(g, {"vae"})
    cs = int(cfg.chunk_size)
    decs = [VAEDecodeLatent(j, cs) for j in range(cs)]
    collect = CollectFrames(cs)
    for op in [*decs, collect]:
        g.add_operator(op)
    g.external_inputs = [TensorPort("chunk_latents")]
    g.external_outputs = [TensorPort("chunk_video")]
    g.chunk_ops = [op.name for op in [*decs, collect]]
    for j in range(1, cs):
        g.add_edge(IREdge(f"vae_decode_{j-1}", None, f"vae_decode_{j}", None))
    for j in range(cs):
        g.add_edge(IREdge(f"vae_decode_{j}", "frames", "collect_frames", f"f{j}"))
    return g


def build_worker_graph(cfg) -> IRGraph:
    """Coarse worker-level graph with COMPOSITE DiT/VAE stages."""
    g = IRGraph(name="worldplay_worker")
    _add_state(g, {"cam", "latents", "kv", "vae", "enc_kv", "first_img"})

    cs = int(cfg.chunk_size)
    ingest = IngestActions()
    dit = DiTStage(int(cfg.num_inference_steps))
    vae = VAEStage(cs)
    vw = VideoWrite()
    for op in [ingest, dit, vae, vw]:
        g.add_operator(op)

    g.external_inputs = [TensorPort("action_codes")]
    g.external_outputs = [TensorPort("chunk_video")]
    g.chunk_ops = ["ingest_actions", "dit_denoise", "vae_decode", "video_write"]
    g.sub_graphs["dit_model"] = _build_dit_subgraph(cfg)
    g.sub_graphs["vae_model"] = _build_vae_subgraph(cfg)

    g.add_edge(IREdge("ingest_actions", "ready", "dit_denoise", "ready"))
    g.add_edge(IREdge("dit_denoise", "chunk_latents", "vae_decode", "chunk_latents",
                      streaming=StreamingInfo(
                          pattern=StreamingPattern.FIXED_RATE, src_chunk_size=cs, dst_chunk_size=1,
                          description="DiT (BATCH) -> VAE (STREAMING) latent-by-latent")))
    g.add_edge(IREdge("vae_decode", "chunk_video", "video_write", "chunk_video"))

    errs = g.validate()
    if errs:
        raise ValueError("worker graph invalid:\n  - " + "\n  - ".join(errs))
    return g
