"""IR operators for the Qwen3-Omni pipeline.

Two graphs are built from these ops (see graph_builder.py):

1. Worker-level graph (coarse pipeline stages):
     thinker (BLACK_BOX) -> talker (COMPOSITE) -> code2wav (BLACK_BOX),
   connected by STREAMING edges. Surfaces the 3 pipeline stages and the
   two streaming overlaps that motivate the pipelined/streamed variants.

2. Talker model-level graph (fine per-codec-frame ops):
     talker_sample_layer0 -> talker_mtp_predict -> talker_decode,
   sharing chunk_persistent state (KV cache, last logits/hidden, sampling
   RNG). Makes the autoregressive cross-frame dependency visible: all
   three ops transitively share persistent state, so find_pipeline_stages
   collapses them into ONE stage -> the talker cannot be pipelined across
   frames; its only model-parallel lever is WITHIN a frame (tensor parallel).

The fine ops replicate the vendored runner's ``_step_unlocked`` exactly,
driving ``ctx.talker.runner`` directly, so the SequentialExecutor produces
bit-identical codec frames to the reference talker (Phase-2 validation).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from wllm.serving.ir import IROperator, OpType, StreamMode, TensorPort


# ===========================================================================
# Context passed to every op's execute()
# ===========================================================================


class OmniContext:
    """Session-wide handles + per-chunk scratch for the IR executor.

    Holds the loaded components (thinker/talker/code2wav) and, during
    talker model-graph execution, the per-frame conditioning embedding.
    """

    def __init__(self, thinker=None, talker=None, code2wav=None,
                 async_runner=None, default_sr: int = 24000,
                 samples_per_frame: int = 1920):
        self.thinker = thinker            # ThinkerComponent
        self.talker = talker              # TalkerComponent (wraps runner)
        self.code2wav = code2wav          # Code2WavComponent
        self.async_runner = async_runner  # asyncio.Runner for engine calls
        self.default_sr = default_sr
        self.samples_per_frame = samples_per_frame
        # per-chunk (talker model graph) scratch
        self.cond = None                  # [1, talker_hidden] conditioning row
        self.talker_eos = False
        # worker-graph scratch
        self.request_id = None


# ===========================================================================
# Worker-level ops (coarse stages)
# ===========================================================================


class ThinkerStage(IROperator):
    """BLACK_BOX thinker: text prompt -> ThinkerOutput (whole)."""

    def __init__(self):
        super().__init__(
            name="thinker",
            op_type=OpType.BLACK_BOX,
            inputs=[TensorPort("user_text")],
            outputs=[TensorPort("thinker_output")],
            state_writes=["thinker_result"],
            stream_mode=StreamMode.STREAMING,
        )

    def execute(self, inputs, ctx: OmniContext, state):
        rid = ctx.request_id or "ir"
        to = ctx.async_runner.run(
            ctx.thinker.run_to_completion(inputs["user_text"], f"{rid}-thinker"))
        state.set("thinker_result", to)
        return {"thinker_output": to}


class TalkerStage(IROperator):
    """COMPOSITE talker: ThinkerOutput -> list[codec_frame]. At the worker
    level this is one stage; its internals are the talker model-level graph."""

    def __init__(self):
        super().__init__(
            name="talker",
            op_type=OpType.COMPOSITE,
            inputs=[TensorPort("thinker_output")],
            outputs=[TensorPort("codec_frames")],
            state_reads=["thinker_result"],
            state_writes=["talker_codec"],
            stream_mode=StreamMode.STREAMING,
            sub_graph="talker_model",
        )

    def execute(self, inputs, ctx: OmniContext, state):
        frames = ctx.talker.run_to_completion(inputs["thinker_output"])
        state.set("talker_codec", frames)
        return {"codec_frames": frames}


class Code2WavStage(IROperator):
    """BLACK_BOX code2wav: list[codec_frame] -> audio waveform (whole)."""

    def __init__(self):
        super().__init__(
            name="code2wav",
            op_type=OpType.BLACK_BOX,
            inputs=[TensorPort("codec_frames")],
            outputs=[TensorPort("audio")],
            state_reads=["talker_codec"],
            stream_mode=StreamMode.STREAMING,
        )

    def execute(self, inputs, ctx: OmniContext, state):
        rid = ctx.request_id or "ir"
        audio, sr = ctx.async_runner.run(
            ctx.code2wav.vocode(inputs["codec_frames"], f"{rid}-c2w", ctx.default_sr))
        return {"audio": (audio, sr)}


# ===========================================================================
# Talker model-level ops (fine per-frame; replicate runner._step_unlocked)
# ===========================================================================


class TalkerPrime(IROperator):
    """Preamble: prime the talker from a ThinkerOutput (prefill + trailing
    queue). Seeds last_logits/last_hidden into chunk_persistent state."""

    def __init__(self):
        super().__init__(
            name="talker_prime",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("thinker_output")],
            outputs=[TensorPort("primed")],
            state_writes=["talker_kv", "talker_last_logits",
                          "talker_last_hidden", "talker_sampling"],
        )

    def execute(self, inputs, ctx: OmniContext, state):
        ctx.talker.prime_whole(inputs["thinker_output"])
        r = ctx.talker.runner
        # seed persistent handles (the runner IS the mutated object)
        state.set("talker_kv", r)
        state.set("talker_last_logits", r._last_logits)
        state.set("talker_last_hidden", r._last_hidden)
        state.set("talker_sampling", r)
        ctx.talker_eos = False
        return {"primed": True}


class TalkerSampleLayer0(IROperator):
    """Sample the first RVQ codec layer from last-position logits.
    Cross-frame: reads talker_last_logits (written by prior frame's decode)."""

    def __init__(self):
        super().__init__(
            name="talker_sample_layer0",
            op_type=OpType.EXPOSED,
            inputs=[],
            outputs=[TensorPort("first_layer_token")],
            state_reads=["talker_last_logits", "talker_sampling"],
            state_writes=["talker_sampling"],
        )

    def execute(self, inputs, ctx: OmniContext, state):
        r = ctx.talker.runner
        _ = state.get("talker_last_logits")   # in-place read of prior-frame logits
        _ = state.get("talker_sampling")
        last_pos_logits = r._last_logits[:, -1, :]
        tok_id = r._sample_first_layer(last_pos_logits, generator=r._sampling_generator)
        first_layer_token = torch.tensor([[tok_id]], dtype=torch.long, device=r.device)
        is_eos = tok_id == r.codec_eos_token_id
        r._sampled_token_history.append(tok_id)
        if is_eos:
            r._codec_eos_seen = True
            ctx.talker_eos = True
        state.set("talker_sampling", r)
        return {"first_layer_token": (first_layer_token, is_eos)}


class TalkerMTPPredict(IROperator):
    """CodePredictor MTP: first-layer token + last hidden + cond -> the 15
    residual RVQ codes and the summed next-talker-input embedding.
    Cross-frame: reads talker_last_hidden (written by prior frame's decode)."""

    def __init__(self):
        super().__init__(
            name="talker_mtp_predict",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("first_layer_token")],
            outputs=[TensorPort("mtp_out")],
            state_reads=["talker_last_hidden"],
        )

    def execute(self, inputs, ctx: OmniContext, state):
        # Always runs (the executor requires downstream external outputs to be
        # produced every chunk); the EOS/None case is handled internally.
        r = ctx.talker.runner
        _ = state.get("talker_last_hidden")
        first_layer_token, is_eos = inputs["first_layer_token"]
        if is_eos:
            return {"mtp_out": None}
        layer0_embed = r.talker.get_input_embeddings()(first_layer_token)
        last_layer_hidden = r._last_hidden[-1][:, -1:].to(
            device=layer0_embed.device, dtype=layer0_embed.dtype)
        cond = ctx.cond.to(device=r.device, dtype=r.dtype).reshape(1, 1, -1)
        next_input_embed, all_token_ids = r.talker.talker_mtp_forward(
            first_layer_token, layer0_embed,
            last_talker_hidden=last_layer_hidden, text_step=cond)
        all_token_ids = all_token_ids.to(torch.long).clone()
        residual_token_ids = all_token_ids[:, 1:]
        return {"mtp_out": (next_input_embed, first_layer_token, residual_token_ids)}


class TalkerDecode(IROperator):
    """Talker MoE forward_decode over the summed embedding -> new logits +
    hidden, writing the paged KV cache in place. Writes the cross-frame
    persistent state (kv, last_logits, last_hidden) the next frame reads."""

    def __init__(self):
        super().__init__(
            name="talker_decode",
            op_type=OpType.EXPOSED,
            inputs=[TensorPort("mtp_out")],
            outputs=[TensorPort("codec_frame")],
            state_reads=["talker_kv"],
            state_writes=["talker_kv", "talker_last_logits", "talker_last_hidden"],
        )

    def execute(self, inputs, ctx: OmniContext, state):
        # Always runs; returns codec_frame=None on the EOS chunk.
        r = ctx.talker.runner
        _ = state.get("talker_kv")   # in-place mutation target (paged KV cache)
        mtp = inputs["mtp_out"]
        if mtp is None:
            return {"codec_frame": None}
        next_input_embed, first_layer_token, residual_token_ids = mtp
        decode_pos_ids = torch.tensor([[r._cache_len]], dtype=torch.long, device=r.device)
        cache_pos_t = torch.tensor(r._cache_len, dtype=torch.long, device=r.device)
        with torch.inference_mode():
            logits, hidden = r.talker.forward_decode(next_input_embed, decode_pos_ids, cache_pos_t)
        r._last_hidden = (hidden,)
        r._last_logits = logits
        r._generation_step += 1
        r._cache_len += 1
        codec_frame = torch.cat([first_layer_token, residual_token_ids], dim=-1
                                ).reshape(-1).to(torch.long).clone()
        state.set("talker_last_logits", logits)
        state.set("talker_last_hidden", (hidden,))
        return {"codec_frame": codec_frame}
