"""IRGraph builders for the LiveAvatar pipeline.

build_model_graph(): the exposed sound-to-video model (one 480 ms audio chunk =
one IR "chunk"). Executable via SequentialExecutor and validated in Phase 2
against the reference LiveAvatarPipeline.step(). Its analysis surfaces the
denoising-step pipeline stages and the VAE stage.

build_worker_graph(): the high-level worker schedule (ASR -> LLM -> TTS ->
LiveAvatar), one IR "chunk" = one utterance. Black-box stages (ASR/LLM/TTS)
are structural only; the streaming edges LLM->TTS and TTS->LiveAvatar encode
the pipeline-scheduling overlap the reference leaves on the table (it
materializes each stage fully before the next).
"""
from __future__ import annotations

from wllm.serving.ir import (
    IRGraph, IROperator, OpType, StateObject, StreamMode, TensorPort, IREdge,
)
from wllm.serving.ir.graph import StreamingInfo, StreamingPattern

from wllm.apps.liveavatar.backend.rocm.ir.ops import (
    DrawNoise, DenoiseStep, VaeDecode,
)


def build_model_graph(num_inference_steps: int = 4) -> IRGraph:
    g = IRGraph(name="liveavatar_model")

    # persistent state
    for k in range(num_inference_steps):
        g.add_state(StateObject(
            name=f"step_{k}_kv", scope="chunk_persistent",
            description=f"DiT KV cache for denoising step {k} (ring over context window)",
        ))
    g.add_state(StateObject(
        name="vae_cache", scope="chunk_persistent",
        description="Causal VAE decoder temporal cache (advances every decoded frame)",
    ))
    g.add_state(StateObject(
        name="ref_latents", scope="session_init",
        description="reference-image latent; set at init, updated once at end of chunk 0",
    ))
    g.add_state(StateObject(
        name="motion_latents", scope="session_init",
        description="motion-prefix latents (framepack condition), set at session init",
    ))

    # ops
    g.add_operator(DrawNoise())
    for k in range(num_inference_steps):
        g.add_operator(DenoiseStep(k, num_inference_steps))
    g.add_operator(VaeDecode())

    # data edges: noise -> step0 -> step1 -> ... -> stepN -> vae
    g.add_edge(IREdge("draw_noise", "noise", "denoise_step_0", "latents"))
    for k in range(num_inference_steps - 1):
        g.add_edge(IREdge(f"denoise_step_{k}", "latents",
                          f"denoise_step_{k + 1}", "latents"))
    g.add_edge(IREdge(f"denoise_step_{num_inference_steps - 1}", "latents",
                      "vae_decode", "latents"))

    g.external_inputs = [TensorPort("audio_features")]
    g.external_outputs = [TensorPort("video")]
    g.preamble_ops = []
    g.chunk_ops = (
        ["draw_noise"]
        + [f"denoise_step_{k}" for k in range(num_inference_steps)]
        + ["vae_decode"]
    )
    errors = g.validate()
    if errors:
        raise ValueError("model graph invalid:\n  " + "\n  ".join(errors))
    return g


# -- worker-level graph (structural; black boxes) ---------------------------

class _Stage(IROperator):
    """Structural stage op (black boxes are not executed in Phase-2 validation)."""

    def __init__(self, name, op_type, inputs, outputs, stream_mode, sub_graph=None):
        super().__init__(name=name, op_type=op_type,
                         inputs=[TensorPort(i) for i in inputs],
                         outputs=[TensorPort(o) for o in outputs],
                         stream_mode=stream_mode, sub_graph=sub_graph)

    def execute(self, inputs, ctx, state):  # structural passthrough
        return {o.name: inputs.get(next(iter(inputs), None)) for o in self.outputs}


def build_worker_graph() -> IRGraph:
    g = IRGraph(name="liveavatar_worker")

    g.add_operator(_Stage("asr", OpType.BLACK_BOX, ["utterance_audio"], ["text"],
                          StreamMode.BATCH))
    g.add_operator(_Stage("llm", OpType.BLACK_BOX, ["text"], ["response_text"],
                          StreamMode.STREAMING))
    g.add_operator(_Stage("tts", OpType.BLACK_BOX, ["response_text"], ["response_audio"],
                          StreamMode.STREAMING))
    g.add_operator(_Stage("liveavatar", OpType.COMPOSITE, ["response_audio"], ["video"],
                          StreamMode.STREAMING, sub_graph="liveavatar_model"))

    # ASR->LLM is batch (LLM needs the full transcript).
    g.add_edge(IREdge("asr", "text", "llm", "text"))
    # LLM streams tokens/sentences to TTS (variable-rate, content-dependent).
    g.add_edge(IREdge("llm", "response_text", "tts", "response_text",
                      streaming=StreamingInfo(
                          pattern=StreamingPattern.VARIABLE_RATE,
                          description="LLM emits tokens; TTS can start on the first sentence",
                      )))
    # TTS streams audio; LiveAvatar re-chunks to 7680-sample (480 ms) DiT chunks.
    g.add_edge(IREdge("tts", "response_audio", "liveavatar", "response_audio",
                      streaming=StreamingInfo(
                          pattern=StreamingPattern.VARIABLE_RATE,
                          src_chunk_size=None, dst_chunk_size=7680,
                          description="TTS audio streamed; LiveAvatar consumes 7680-sample chunks -> 24 frames each",
                      )))

    g.external_inputs = [TensorPort("utterance_audio")]
    g.external_outputs = [TensorPort("video")]
    g.preamble_ops = []
    g.chunk_ops = ["asr", "llm", "tts", "liveavatar"]
    g.sub_graphs = {"liveavatar_model": build_model_graph()}
    errors = g.validate()
    if errors:
        raise ValueError("worker graph invalid:\n  " + "\n  ".join(errors))
    return g


if __name__ == "__main__":
    from wllm.serving.ir import summarize_graph
    print(summarize_graph(build_model_graph()))
    print("\n\n########## WORKER ##########\n")
    print(summarize_graph(build_worker_graph()))
