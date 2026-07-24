"""Build the Qwen3-Omni IR graphs.

* ``build_worker_graph`` — the high-level 3-stage pipeline (Thinker ->
  Talker -> Code2Wav) with STREAMING edges. One "chunk" = one prompt;
  there is no cross-prompt persistent data state (the talker session is
  reset per prompt), so the three stages are cross-chunk independent and
  ``find_pipeline_stages`` reports them as separate stages = candidates
  for pipeline parallelism across GPUs. ``find_streaming_overlaps``
  surfaces the thinker->talker (variable rate) and talker->code2wav
  (fixed rate, 1 codec frame -> 25-frame vocoder chunk) overlaps.

* ``build_talker_model_graph`` — the EXPOSED Talker decomposed into
  per-codec-frame operators. One "chunk" = one codec frame. All three
  ops share the chunk-persistent talker state (KV cache, last_hidden,
  last_logits, history, positions), so ``analyze_cross_chunk_dependencies``
  reports zero independent pairs: the talker is a strict sequential chain
  across frames and cannot be pipelined frame-to-frame. Its only
  model-parallel lever is within one frame's forward (tensor parallelism),
  which is below the IR operator granularity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from wllm.serving.ir import (
    IRGraph, IREdge, StateObject, TensorPort,
    StreamingInfo, StreamingPattern,
)

from wllm.apps.qwen3_omni.backend.cuda.ir.ops import (
    ThinkerOp, TalkerOp, Code2WavOp,
    TalkerSampleFirstLayer, TalkerMTP, TalkerDecode,
)


@dataclass
class PipelineContext:
    """Opaque context handed to every operator's execute()."""
    cfg: Any
    thinker_engine: Any = None
    talker_runner: Any = None
    c2w_engine: Any = None
    async_runner: Any = None
    request_id: str = "ir"
    # talker-model-graph per-frame control flags (set by ops):
    is_eos: bool = False
    last_frame: Any = None
    last_sample_rate: Optional[int] = None


# ----------------------------------------------------------------------
# Worker-level graph
# ----------------------------------------------------------------------

def build_worker_graph() -> IRGraph:
    g = IRGraph(name="qwen3_omni_worker")
    g.add_operator(ThinkerOp())
    g.add_operator(TalkerOp())
    g.add_operator(Code2WavOp())

    g.external_inputs = [TensorPort("user_text")]
    g.external_outputs = [TensorPort("audio")]

    # thinker -> talker : the thinker emits decode tokens incrementally;
    # the talker can consume them as they arrive. Content-dependent #tokens
    # -> VARIABLE_RATE.
    g.add_edge(IREdge(
        src_op="thinker", src_port="thinker_out",
        dst_op="talker", dst_port="thinker_out",
        streaming=StreamingInfo(
            pattern=StreamingPattern.VARIABLE_RATE,
            description=("thinker decode tokens stream into the talker trailing "
                         "queue; talker stalls if it outpaces the thinker"),
        ),
    ))
    # talker -> code2wav : talker emits 1 codec frame (12.5 Hz) per step;
    # code2wav vocodes fixed-size chunks (codec_chunk_frames, default 25)
    # with left context. FIXED_RATE 1 -> 25.
    g.add_edge(IREdge(
        src_op="talker", src_port="codec_frames",
        dst_op="code2wav", dst_port="codec_frames",
        streaming=StreamingInfo(
            pattern=StreamingPattern.FIXED_RATE,
            src_chunk_size=1, dst_chunk_size=25,
            description=("each codec frame is 1920 samples (80 ms @ 24 kHz); "
                         "code2wav vocodes accumulating chunks of 25 frames"),
        ),
    ))

    g.chunk_ops = ["thinker", "talker", "code2wav"]
    g.preamble_ops = []
    # Link the talker's per-frame sub-graph so the COMPOSITE reference resolves.
    g.sub_graphs["talker_model"] = build_talker_model_graph()
    errors = g.validate()
    if errors:
        raise ValueError(f"worker graph invalid: {errors}")
    return g


# ----------------------------------------------------------------------
# Talker model-level graph
# ----------------------------------------------------------------------

def build_talker_model_graph() -> IRGraph:
    g = IRGraph(name="qwen3_omni_talker_model")
    g.add_operator(TalkerSampleFirstLayer())
    g.add_operator(TalkerMTP())
    g.add_operator(TalkerDecode())

    for name, desc in [
        ("talker_kv", "talker paged KV cache + sampling RNG (the runner)"),
        ("last_logits", "[1,T,vocab] logits from previous talker forward"),
        ("last_hidden", "[1,T,hidden] hidden from previous talker forward"),
        ("sampled_history", "list[int] of emitted layer-0 tokens (rep. penalty)"),
        ("gen_step", "int: codec frames emitted so far (selects text cond)"),
        ("cache_len", "int: next KV write position"),
    ]:
        g.add_state(StateObject(name=name, description=desc, scope="chunk_persistent"))

    # within-frame data chain
    g.add_edge(IREdge("talker_sample_first_layer", "first_token", "talker_mtp", "first_token"))
    g.add_edge(IREdge("talker_sample_first_layer", "layer0_embed", "talker_mtp", "layer0_embed"))
    g.add_edge(IREdge("talker_sample_first_layer", "cond", "talker_mtp", "cond"))
    g.add_edge(IREdge("talker_mtp", "next_input_embed", "talker_decode", "next_input_embed"))
    g.add_edge(IREdge("talker_mtp", "residual_codes", "talker_decode", "residual_codes"))
    g.add_edge(IREdge("talker_sample_first_layer", "first_token", "talker_decode", "first_token"))

    g.chunk_ops = ["talker_sample_first_layer", "talker_mtp", "talker_decode"]
    g.preamble_ops = []
    # codec_frame is delivered via ctx.last_frame (decode is skipped on EOS,
    # so it is not a mandatory external output).
    g.external_outputs = []
    errors = g.validate()
    if errors:
        raise ValueError(f"talker model graph invalid: {errors}")
    return g


def seed_talker_state_from_runner(executor, runner) -> None:
    """Pre-populate the talker-model-graph executor state from a freshly
    primed runner (after blocks.prime_talker / start_session)."""
    executor.init_state({
        "talker_kv": runner,
        "last_logits": runner._last_logits,
        "last_hidden": runner._last_hidden,
        "sampled_history": list(runner._sampled_token_history),
        "gen_step": runner._generation_step,
        "cache_len": runner._cache_len,
    })
