"""IR operators for the Qwen3-Omni pipeline.

Two families:

* **Worker-level ops** (`ThinkerOp`, `TalkerOp`, `Code2WavOp`) — the three
  high-level stages. Thinker and Code2Wav are BLACK_BOX (vLLM-Omni
  engines); Talker is EXPOSED (in-process runner, decomposed below).
  Connected by STREAMING edges: thinker emits text tokens incrementally
  (variable rate), talker emits codec frames at a fixed 12.5 Hz that
  code2wav vocodes in fixed-size chunks.

* **Talker model-level ops** (`TalkerSampleFirstLayer`, `TalkerMTP`,
  `TalkerDecode`) — a faithful decomposition of
  Qwen3OmniTalkerRunner._step_unlocked() into three operators so the IR
  exposes the talker's per-frame state dependencies. They drive the
  runner's own sub-methods (so the numerics are identical to step()) but
  thread the per-frame state (last_logits, last_hidden, history, cache
  position) through the IR StateStore for dependency analysis. The talker
  KV cache + sampling RNG stay inside the runner and are declared as the
  `talker_kv` chunk-persistent state (in-place mutation via state_writes).

The context object (see graph_builder.PipelineContext) carries the cfg,
the engines, the talker runner, and an asyncio.Runner.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import torch

from wllm.serving.ir import IROperator, OpType, StreamMode, TensorPort

from wllm.apps.qwen3_omni.backend.cuda import blocks


# ======================================================================
# Worker-level operators
# ======================================================================

class ThinkerOp(IROperator):
    """BLACK_BOX vLLM-Omni Thinker. text -> ThinkerOutput (streaming text)."""

    def __init__(self) -> None:
        super().__init__(
            name="thinker",
            op_type=OpType.BLACK_BOX,
            inputs=[TensorPort("user_text")],
            outputs=[TensorPort("thinker_out")],
            stream_mode=StreamMode.STREAMING,
        )

    def execute(self, inputs: dict, ctx: Any, state) -> dict:
        rid = f"{ctx.request_id}-thinker"
        out = blocks.run_thinker(ctx.thinker_engine, ctx.cfg, inputs["user_text"], rid,
                                 runner=ctx.async_runner)
        return {"thinker_out": out}


class TalkerOp(IROperator):
    """EXPOSED in-process Talker. ThinkerOutput -> list[codec_frame].

    Coarse worker-level view; the per-frame internals are the talker
    model-level graph (sub_graph='talker_model')."""

    def __init__(self) -> None:
        super().__init__(
            name="talker",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("thinker_out")],
            outputs=[TensorPort("codec_frames")],
            stream_mode=StreamMode.STREAMING,
            sub_graph="talker_model",
        )

    def execute(self, inputs: dict, ctx: Any, state) -> dict:
        runner = ctx.talker_runner
        blocks.prime_talker(runner, inputs["thinker_out"], ctx.cfg, push_all=True)
        frames = blocks.run_talker_to_completion(runner)
        return {"codec_frames": frames}


class Code2WavOp(IROperator):
    """BLACK_BOX vLLM-Omni vocoder. list[codec_frame] -> audio samples."""

    def __init__(self) -> None:
        super().__init__(
            name="code2wav",
            op_type=OpType.BLACK_BOX,
            inputs=[TensorPort("codec_frames")],
            outputs=[TensorPort("audio")],
            stream_mode=StreamMode.STREAMING,
        )

    def execute(self, inputs: dict, ctx: Any, state) -> dict:
        rid = f"{ctx.request_id}-c2w"
        audio, sr = blocks.vocode_full(ctx.c2w_engine, ctx.cfg, inputs["codec_frames"], rid,
                                       runner=ctx.async_runner)
        ctx.last_sample_rate = sr
        return {"audio": audio}


# ======================================================================
# Talker model-level operators (decompose _step_unlocked)
# ======================================================================
#
# Per-frame state, all chunk_persistent (each codec frame depends on the
# previous one through these):
#   talker_kv      : the runner (paged KV cache + RNG), mutated in place
#   last_logits    : [1, T, vocab] from the previous forward
#   last_hidden    : [1, T, hidden] from the previous forward
#   sampled_history: list[int] of emitted layer-0 tokens (repetition pen.)
#   gen_step       : int, frames emitted so far (selects the text cond)
#   cache_len      : int, next KV write position
#
# These three ops form a strict within-frame chain AND share the above
# persistent state across frames, so analyze_cross_chunk_dependencies
# reports zero independent pairs -> the talker cannot be pipelined across
# frames; its only model-parallel lever is within-frame (TP), which lives
# below the IR's operator granularity.


def _select_cond(runner, gen_step: int):
    """Replicate step()'s text-conditioning selection for frame `gen_step`."""
    trailing = runner._trailing_decode_embeds
    n_decode = len(trailing)
    if gen_step < n_decode:
        return trailing[gen_step]
    if runner._thinker_session_finished:
        if gen_step == n_decode:
            return runner._tts_eos_embed
        return runner._tts_pad_embed
    return None  # would stall (queue empty, thinker not finished)


class TalkerSampleFirstLayer(IROperator):
    def __init__(self) -> None:
        super().__init__(
            name="talker_sample_first_layer",
            op_type=OpType.EXPOSED,
            inputs=[],
            outputs=[TensorPort("first_token"), TensorPort("layer0_embed"),
                     TensorPort("cond"), TensorPort("is_eos")],
            state_reads=["talker_kv", "last_logits", "sampled_history", "gen_step"],
            state_writes=["sampled_history"],
            stream_mode=StreamMode.STREAMING,
        )

    def execute(self, inputs: dict, ctx: Any, state) -> dict:
        runner = state.get("talker_kv")
        last_logits = state.get("last_logits")
        history = state.get("sampled_history")
        gen_step = state.get("gen_step")
        # Make the runner see the IR-tracked history for repetition penalty.
        runner._sampled_token_history = history
        last_pos_logits = last_logits[:, -1, :]
        first_id = runner._sample_first_layer(last_pos_logits,
                                              generator=runner._sampling_generator)
        is_eos = (first_id == runner.codec_eos_token_id)
        ctx.is_eos = bool(is_eos)
        first_token = torch.tensor([[first_id]], dtype=torch.long, device=runner.device)
        if is_eos:
            history.append(first_id)
            state.set("sampled_history", history)
            return {"first_token": first_token, "layer0_embed": None,
                    "cond": None, "is_eos": True}
        layer0_embed = runner.talker.get_input_embeddings()(first_token)
        cond = _select_cond(runner, gen_step)
        if cond is None:
            raise RuntimeError("talker cond underrun: queue empty, thinker not finished")
        state.set("sampled_history", history)
        return {"first_token": first_token, "layer0_embed": layer0_embed,
                "cond": cond, "is_eos": False}


class TalkerMTP(IROperator):
    def __init__(self) -> None:
        super().__init__(
            name="talker_mtp",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("first_token"), TensorPort("layer0_embed"),
                    TensorPort("cond")],
            outputs=[TensorPort("next_input_embed"), TensorPort("residual_codes")],
            state_reads=["talker_kv", "last_hidden"],
            state_writes=[],
            stream_mode=StreamMode.STREAMING,
        )

    def should_run(self, context: Any) -> bool:
        return not getattr(context, "is_eos", False)

    def execute(self, inputs: dict, ctx: Any, state) -> dict:
        runner = state.get("talker_kv")
        last_hidden = state.get("last_hidden")
        first_token = inputs["first_token"]
        layer0_embed = inputs["layer0_embed"]
        cond = inputs["cond"]
        last_layer_hidden = last_hidden[-1][:, -1:].to(
            device=layer0_embed.device, dtype=layer0_embed.dtype)
        cond_dev = cond.to(device=runner.device, dtype=runner.dtype).reshape(1, 1, -1)
        next_input_embed, all_token_ids = runner.talker.talker_mtp_forward(
            first_token, layer0_embed,
            last_talker_hidden=last_layer_hidden, text_step=cond_dev)
        all_token_ids = all_token_ids.to(torch.long).clone()
        residual = all_token_ids[:, 1:]
        return {"next_input_embed": next_input_embed, "residual_codes": residual}


class TalkerDecode(IROperator):
    def __init__(self) -> None:
        super().__init__(
            name="talker_decode",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("first_token"), TensorPort("next_input_embed"),
                    TensorPort("residual_codes")],
            outputs=[TensorPort("codec_frame")],
            state_reads=["talker_kv"],
            state_writes=["talker_kv", "last_hidden", "last_logits",
                          "gen_step", "cache_len", "sampled_history"],
            stream_mode=StreamMode.STREAMING,
        )

    def should_run(self, context: Any) -> bool:
        return not getattr(context, "is_eos", False)

    def execute(self, inputs: dict, ctx: Any, state) -> dict:
        runner = state.get("talker_kv")
        cache_len = state.get("cache_len")
        gen_step = state.get("gen_step")
        history = state.get("sampled_history")
        next_input_embed = inputs["next_input_embed"]
        first_token = inputs["first_token"]
        residual = inputs["residual_codes"]
        decode_pos = torch.tensor([[cache_len]], dtype=torch.long, device=runner.device)
        cache_pos = torch.tensor(cache_len, dtype=torch.long, device=runner.device)
        with torch.inference_mode():
            logits, hidden = runner.talker.forward_decode(
                next_input_embed, decode_pos, cache_pos)
        state.set("last_hidden", (hidden,))
        state.set("last_logits", logits)
        history.append(int(first_token.item()))
        state.set("sampled_history", history)
        state.set("gen_step", gen_step + 1)
        state.set("cache_len", cache_len + 1)
        frame = torch.cat([first_token, residual], dim=-1).reshape(-1).to(torch.long).clone()
        ctx.last_frame = frame
        return {"codec_frame": frame}
