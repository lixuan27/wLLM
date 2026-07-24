"""Build the WorldPlay IR graphs.

- `build_chunk_graph()` : the fine-grained, executable model-level graph
  (camera -> select_mem -> kv_fill -> denoise_0..3 -> finalize -> vae_0..3 ->
  collect). This is the graph that was validated numerically against the
  reference.
- `build_worker_graph()` : a coarse worker-level view (poll -> camera -> DiT
  (composite) -> VAE (composite) -> write) for the high-level streaming /
  device-placement analysis.
"""

from __future__ import annotations

from wllm.serving.ir import (
    IRGraph, IREdge, OpType, StateObject, StreamMode, StreamingInfo,
    StreamingPattern, TensorPort, IROperator, ir_operator,
)

from wllm.apps.worldplay.backend.rocm.ir import ops as O


def _persistent(name, desc):
    return StateObject(name=name, description=desc, scope="chunk_persistent")


def build_chunk_graph(num_denoise: int = 4, num_latents: int = 4) -> IRGraph:
    """Fine-grained per-chunk graph. `num_denoise`=inference steps (4),
    `num_latents`=chunk_size (4)."""
    g = IRGraph(name="worldplay_chunk")

    # --- persistent state ---
    g.add_state(_persistent(O.S_CAMERA, "camera pose accumulators T / C_inv"))
    g.add_state(_persistent(O.S_COND, "viewmats / Ks / action accumulators"))
    g.add_state(_persistent(O.S_LATENTS, "fp32 latent store (cross-chunk context window)"))
    g.add_state(_persistent(O.S_KV, "DiT prope KV cache"))
    g.add_state(_persistent(O.S_VAE, "VAE temporal causal cache"))

    # --- preamble ---
    g.add_operator(O.session_init)
    g.preamble_ops = ["session_init"]

    # --- chunk operators ---
    g.add_operator(O.camera_decode)
    g.add_operator(O.prep)
    g.add_operator(O.select_mem)
    g.add_operator(O.kv_fill)
    denoise = [O.DenoiseStep(i) for i in range(num_denoise)]
    for d in denoise:
        g.add_operator(d)
    g.add_operator(O.finalize)
    vae = [O.VaeDecode(j) for j in range(num_latents)]
    for v in vae:
        g.add_operator(v)
    collect = O.CollectFrames(num_latents)
    g.add_operator(collect)

    g.chunk_ops = (
        ["camera_decode", "prep", "select_mem", "kv_fill"]
        + [f"denoise_{i}" for i in range(num_denoise)]
        + ["finalize"]
        + [f"vae_decode_{j}" for j in range(num_latents)]
        + ["collect_frames"]
    )

    g.external_inputs = [TensorPort("actions", ("chunk_size",), "int64")]
    g.external_outputs = [TensorPort("chunk_frames", ("F", "H", "W", 3), "uint8")]

    # --- data edges: camera -> prep ---
    for port in ("viewmats", "Ks", "action"):
        g.add_edge(IREdge("camera_decode", port, "prep", port))

    # --- ordering edges through the DiT stage ---
    g.add_edge(IREdge("prep", None, "select_mem", None))
    g.add_edge(IREdge("prep", None, "kv_fill", None))
    g.add_edge(IREdge("select_mem", None, "kv_fill", None))
    g.add_edge(IREdge("prep", None, "denoise_0", None))
    g.add_edge(IREdge("kv_fill", None, "denoise_0", None))
    for i in range(1, num_denoise):
        g.add_edge(IREdge(f"denoise_{i-1}", None, f"denoise_{i}", None))
    g.add_edge(IREdge(f"denoise_{num_denoise-1}", None, "finalize", None))

    # --- data edge: finalized latents -> VAE (decoupled from `latents` state,
    #     so the VAE stage is independent of the next chunk's DiT) ---
    for j in range(num_latents):
        g.add_edge(IREdge("finalize", "chunk_latents", f"vae_decode_{j}", "chunk_latents"))
    # VAE temporal-cache chain (each decode depends on the previous)
    for j in range(1, num_latents):
        g.add_edge(IREdge(f"vae_decode_{j-1}", None, f"vae_decode_{j}", None))
    # frames -> collect
    for j in range(num_latents):
        g.add_edge(IREdge(f"vae_decode_{j}", f"frames_{j}", "collect_frames", f"frames_{j}"))

    return g


def build_worker_graph(chunk_size: int = 4, temporal_up: int = 4) -> IRGraph:
    """Coarse worker-level graph for the high-level streaming / placement view.
    All stages are EXPOSED (WorldPlay has no black-box engine)."""
    g = IRGraph(name="worldplay_worker")

    g.add_state(_persistent(O.S_CAMERA, "camera pose accumulators"))
    g.add_state(_persistent(O.S_COND, "conditioning accumulators"))
    g.add_state(_persistent(O.S_LATENTS, "fp32 latent store"))
    g.add_state(_persistent(O.S_KV, "DiT prope KV cache"))
    g.add_state(_persistent(O.S_VAE, "VAE temporal causal cache"))

    @ir_operator(name="poll_actions", op_type=OpType.EXPOSED,
                 inputs=[TensorPort("action_buffer")], outputs=[TensorPort("actions")],
                 state_reads=[], state_writes=[])
    def poll_actions(inputs, ctx, state):
        return {"actions": inputs["action_buffer"]}

    @ir_operator(name="camera", op_type=OpType.EXPOSED,
                 inputs=[TensorPort("actions")], outputs=[TensorPort("cond")],
                 state_reads=[O.S_CAMERA], state_writes=[O.S_CAMERA, O.S_COND])
    def camera(inputs, ctx, state):
        return {"cond": None}

    class DiTStage(IROperator):
        def __init__(self):
            super().__init__(name="dit_step", op_type=OpType.COMPOSITE,
                             inputs=[TensorPort("cond")], outputs=[TensorPort("chunk_latents")],
                             state_reads=[O.S_LATENTS, O.S_KV, O.S_COND],
                             state_writes=[O.S_LATENTS, O.S_KV],
                             stream_mode=StreamMode.BATCH,)
        def execute(self, inputs, ctx, state):
            return {"chunk_latents": None}

    class VaeStage(IROperator):
        def __init__(self):
            super().__init__(name="vae_step", op_type=OpType.COMPOSITE,
                             inputs=[TensorPort("chunk_latents")], outputs=[TensorPort("frames")],
                             state_reads=[O.S_VAE], state_writes=[O.S_VAE],
                             stream_mode=StreamMode.STREAMING,)
        def execute(self, inputs, ctx, state):
            return {"frames": None}

    @ir_operator(name="write_video", op_type=OpType.EXPOSED,
                 inputs=[TensorPort("frames")], outputs=[],
                 state_reads=[], state_writes=[])
    def write_video(inputs, ctx, state):
        return {}

    g.add_operator(poll_actions)
    g.add_operator(camera)
    g.add_operator(DiTStage())
    g.add_operator(VaeStage())
    g.add_operator(write_video)
    g.chunk_ops = ["poll_actions", "camera", "dit_step", "vae_step", "write_video"]

    g.external_inputs = [TensorPort("action_buffer")]
    g.external_outputs = [TensorPort("frames")]

    g.add_edge(IREdge("poll_actions", "actions", "camera", "actions"))
    g.add_edge(IREdge("camera", "cond", "dit_step", "cond"))
    # DiT -> VAE handoff is streaming: chunk_size latents expand to
    # chunk_size * temporal_up pixel frames (rate conversion), and VAE(N) can
    # overlap DiT(N+1) because they share no persistent state.
    g.add_edge(IREdge(
        "dit_step", "chunk_latents", "vae_step", "chunk_latents",
        streaming=StreamingInfo(
            pattern=StreamingPattern.FIXED_RATE,
            src_chunk_size=chunk_size,
            dst_chunk_size=chunk_size * temporal_up,
            description="each finalized latent decodes to temporal_up pixel frames",
        ),
    ))
    g.add_edge(IREdge("vae_step", "frames", "write_video", "frames"))
    return g
