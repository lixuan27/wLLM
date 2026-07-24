"""Build IR graphs for the LiveAvatar app.

Two graphs:
  - build_model_graph(cfg): the EXPOSED, chunk-periodic sound-to-video graph
    (wav2vec -> 4 DiT denoising steps -> VAE decode). Executable via
    SequentialExecutor; validated bit-exact against the reference
    pipeline. This is the graph whose state analysis surfaces the
    per-step pipeline.
  - build_worker_graph(cfg): the high-level worker schedule
    (ASR -> LLM -> TTS -> LiveAvatar), with the black-box engines and the
    streaming edges. Structural only (the black boxes are not executed); used
    for streaming-overlap analysis.
"""
from __future__ import annotations

from wllm.serving.ir import (
    IRGraph, IROperator, IREdge, StateObject, TensorPort, OpType, StreamMode,
    StreamingInfo, StreamingPattern,
)
from wllm.apps.liveavatar.backend.cuda.ir.ops import (
    Wav2VecExtract, DiTDenoiseStep, VAEDecode,
)


def build_model_graph(cfg) -> IRGraph:
    n_steps = int(cfg.num_inference_steps)
    g = IRGraph(name="liveavatar_model")

    # ---- state objects ----
    for k in range(n_steps):
        g.add_state(StateObject(
            name=f"cache_{k}", scope="chunk_persistent",
            description=f"private KV cache for denoising step {k} "
                        f"(self-attn KV accumulates across chunks)"))
    g.add_state(StateObject(
        name="vae_cache", scope="chunk_persistent",
        description="causal VAE decoder temporal cache (in-place mutated)"))
    g.add_state(StateObject(
        name="ref_latents", scope="session_init",
        description="reference-image latent for cond prefill; rebound once on chunk 0"))
    g.add_state(StateObject(
        name="motion_latents", scope="session_init",
        description="motion-prefix latents for cond prefill"))

    # ---- operators ----
    wav2vec = Wav2VecExtract()
    g.add_operator(wav2vec)
    steps = []
    for k in range(n_steps):
        op = DiTDenoiseStep(k)
        g.add_operator(op)
        steps.append(op)
    vae = VAEDecode()
    g.add_operator(vae)

    g.chunk_ops = [wav2vec.name] + [s.name for s in steps] + [vae.name]
    g.external_inputs = [TensorPort("audio_samples"), TensorPort("latents")]
    g.external_outputs = [TensorPort("video")]

    # ---- edges ----
    # wav2vec features fan out to every denoising step (each step re-conditions
    # on the same audio in the reference)
    for s in steps:
        g.add_edge(IREdge(
            src_op=wav2vec.name, src_port="audio_features",
            dst_op=s.name, dst_port="audio_features",
            streaming=StreamingInfo(
                pattern=StreamingPattern.FIXED_RATE,
                src_chunk_size=1, dst_chunk_size=1,
                description="audio features for this chunk, shared by all steps")))
    # latent chain across steps (within-chunk data dependency)
    for i in range(len(steps) - 1):
        g.add_edge(IREdge(
            src_op=steps[i].name, src_port="latents",
            dst_op=steps[i + 1].name, dst_port="latents"))
    # last step -> VAE decode
    g.add_edge(IREdge(
        src_op=steps[-1].name, src_port="latents",
        dst_op=vae.name, dst_port="latents",
        streaming=StreamingInfo(
            pattern=StreamingPattern.FIXED_RATE,
            src_chunk_size=int(cfg.chunk_size), dst_chunk_size=int(cfg.chunk_size),
            description="final-step latents decoded frame-by-frame")))

    errors = g.validate()
    if errors:
        raise ValueError("model graph invalid:\n  - " + "\n  - ".join(errors))
    return g


def build_worker_graph(cfg) -> IRGraph:
    """High-level worker schedule, structural only (black boxes not executed)."""
    g = IRGraph(name="liveavatar_worker")

    asr = IROperator(name="asr", op_type=OpType.BLACK_BOX,
                     inputs=[TensorPort("mic_audio")], outputs=[TensorPort("text")],
                     stream_mode=StreamMode.BATCH)
    llm = IROperator(name="llm", op_type=OpType.BLACK_BOX,
                     inputs=[TensorPort("text")], outputs=[TensorPort("response")],
                     stream_mode=StreamMode.STREAMING)
    tts = IROperator(name="tts", op_type=OpType.BLACK_BOX,
                     inputs=[TensorPort("response")], outputs=[TensorPort("audio")],
                     stream_mode=StreamMode.STREAMING)
    live = IROperator(name="liveavatar", op_type=OpType.COMPOSITE,
                      inputs=[TensorPort("audio")],
                      outputs=[TensorPort("video"), TensorPort("out_audio")],
                      stream_mode=StreamMode.STREAMING, sub_graph="liveavatar_model")
    for op in (asr, llm, tts, live):
        g.add_operator(op)
    g.chunk_ops = ["asr", "llm", "tts", "liveavatar"]
    g.external_inputs = [TensorPort("mic_audio")]
    g.external_outputs = [TensorPort("video"), TensorPort("out_audio")]
    g.sub_graphs = {"liveavatar_model": build_model_graph(cfg)}

    g.add_edge(IREdge(src_op="asr", src_port="text", dst_op="llm", dst_port="text"))
    # LLM streams tokens; TTS consumes sentence-ish spans -> variable-rate overlap
    g.add_edge(IREdge(
        src_op="llm", src_port="response", dst_op="tts", dst_port="response",
        streaming=StreamingInfo(
            pattern=StreamingPattern.VARIABLE_RATE,
            description="LLM token stream -> TTS consumes growing text "
                        "(sentence-granular); overlap legal, ratio content-dependent")))
    # TTS streams audio; LiveAvatar consumes fixed 7680-sample (tts_chunk_size) windows
    g.add_edge(IREdge(
        src_op="tts", src_port="audio", dst_op="liveavatar", dst_port="audio",
        streaming=StreamingInfo(
            pattern=StreamingPattern.FIXED_RATE,
            src_chunk_size=None, dst_chunk_size=int(cfg.tts_chunk_size),
            description="TTS audio stream -> LiveAvatar consumes one "
                        f"{int(cfg.tts_chunk_size)}-sample window per DiT chunk")))

    errors = g.validate()
    if errors:
        raise ValueError("worker graph invalid:\n  - " + "\n  - ".join(errors))
    return g
