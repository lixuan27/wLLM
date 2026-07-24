"""Build the Qwen3-Omni IR graphs.

- ``build_worker_graph()``: coarse pipeline (thinker -> talker -> code2wav)
  with streaming edges. No shared chunk_persistent state across the three
  stages => find_pipeline_stages yields 3 stages (3 GPUs) and
  find_streaming_overlaps yields the 2 streaming overlaps that motivate the
  pipelined/streamed variants.

- ``build_talker_model_graph()``: fine per-codec-frame talker ops. All three
  ops transitively share chunk_persistent state (KV cache + last logits/
  hidden), so find_pipeline_stages collapses them into ONE stage — the
  talker is strictly autoregressive across frames and can only be
  parallelized WITHIN a frame (tensor parallel), not across frames.
"""

from __future__ import annotations

from wllm.serving.ir import (
    IRGraph, IREdge, StateObject, TensorPort,
    StreamingInfo, StreamingPattern,
)

from wllm.apps.qwen3_omni.backend.rocm.ir.ops import (
    ThinkerStage, TalkerStage, Code2WavStage,
    TalkerPrime, TalkerSampleLayer0, TalkerMTPPredict, TalkerDecode,
)


def build_talker_model_graph() -> IRGraph:
    g = IRGraph(name="talker_model")

    # chunk_persistent state = the autoregressive spine of the talker.
    g.add_state(StateObject("talker_kv", "paged KV cache (grows per frame)",
                            scope="chunk_persistent"))
    g.add_state(StateObject("talker_last_logits", "logits from prior frame's decode",
                            scope="chunk_persistent"))
    g.add_state(StateObject("talker_last_hidden", "hidden from prior frame's decode",
                            scope="chunk_persistent"))
    g.add_state(StateObject("talker_sampling", "layer-0 sampling RNG + token history",
                            scope="chunk_persistent"))

    prime = TalkerPrime()
    sample = TalkerSampleLayer0()
    mtp = TalkerMTPPredict()
    decode = TalkerDecode()
    for op in (prime, sample, mtp, decode):
        g.add_operator(op)

    g.preamble_ops.append(prime.name)
    g.chunk_ops.extend([sample.name, mtp.name, decode.name])

    # within-frame data chain
    g.add_edge(IREdge("talker_sample_layer0", "first_layer_token",
                      "talker_mtp_predict", "first_layer_token"))
    g.add_edge(IREdge("talker_mtp_predict", "mtp_out", "talker_decode", "mtp_out"))

    # external I/O for the chunk phase
    g.external_inputs = [TensorPort("cond", ("1", "talker_hidden"))]
    g.external_outputs = [TensorPort("codec_frame", ("16",))]
    return g


def build_worker_graph() -> IRGraph:
    g = IRGraph(name="worker")

    # per-request (not chunk_persistent) state -> no cross-stage coupling.
    g.add_state(StateObject("thinker_result", "ThinkerOutput for this request",
                            scope="ephemeral"))
    g.add_state(StateObject("talker_codec", "codec frame list for this request",
                            scope="ephemeral"))

    thinker = ThinkerStage()
    talker = TalkerStage()
    code2wav = Code2WavStage()
    for op in (thinker, talker, code2wav):
        g.add_operator(op)
    g.chunk_ops.extend([thinker.name, talker.name, code2wav.name])

    # data edges + streaming metadata
    g.add_edge(IREdge(
        "thinker", "thinker_output", "talker", "thinker_output",
        streaming=StreamingInfo(
            pattern=StreamingPattern.VARIABLE_RATE,
            description=("thinker decode-token embeds stream into the talker "
                         "trailing queue (~1 thinker token : 1 codec frame); "
                         "talker consumes embed[gen_step] per frame"))))
    g.add_edge(IREdge(
        "talker", "codec_frames", "code2wav", "codec_frames",
        streaming=StreamingInfo(
            pattern=StreamingPattern.FIXED_RATE,
            src_chunk_size=1, dst_chunk_size=1920,
            description=("1 codec frame -> 1920 audio samples (12.5 Hz codec "
                         "-> 24 kHz); c2w vocodes chunks with left context"))))

    g.external_inputs = [TensorPort("user_text")]
    g.external_outputs = [TensorPort("audio")]

    # attach the talker model graph as the talker stage's sub-graph
    g.sub_graphs["talker_model"] = build_talker_model_graph()
    return g


if __name__ == "__main__":
    from wllm.serving.ir import summarize_graph
    wg = build_worker_graph()
    print("=== WORKER GRAPH ===")
    print("validate:", wg.validate() or "OK")
    print(summarize_graph(wg))
    print("\n=== TALKER MODEL GRAPH ===")
    tg = build_talker_model_graph()
    print("validate:", tg.validate() or "OK")
    print(summarize_graph(tg))
