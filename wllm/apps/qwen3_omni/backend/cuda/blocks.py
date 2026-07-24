"""Pipeline building blocks for Qwen3-Omni, shared by the IR conversion
(Phase 1-2) and the deployment variants (Phase 3-4).

These wrap the THREE reference compute stages so that both the IR
operators and the optimized workers exercise the *identical* numerics as
the user reference backend (wllm/apps/qwen3_omni/reference/worker.py). To stay
provably faithful for Phase-2 validation we reuse the reference's
module-level helpers (chat formatting, thinker-output coalescing,
code2wav prompt packing, audio extraction) and the reference Talker
runner verbatim. Variants that need to change a stage's internals (e.g.
Talker tensor parallelism) vendor + modify their own copy instead.

Stage map:
  * Thinker  : vLLM-Omni AsyncOmni engine (BLACK_BOX). run_thinker ->
               ThinkerOutput (tokens + per-token thinker embed/hidden
               tables + tts markers).
  * Talker   : in-process Qwen3OmniTalkerRunner (EXPOSED). prime_talker
               primes a session from a ThinkerOutput; the runner then
               emits one codec frame per step().
  * Code2Wav : vLLM-Omni AsyncOmni engine (BLACK_BOX). vocode_full
               (single request, reference behavior) or vocode_chunk
               (streaming, with left context).
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

import numpy as np
import torch

from vllm_omni import AsyncOmni
from vllm.sampling_params import SamplingParams

from wllm.serving.logger import init_logger
from wllm.apps.qwen3_omni.reference.config import Qwen3OmniReferenceConfig
from wllm.apps.qwen3_omni.reference.runner import Qwen3OmniTalkerRunner
# Reuse the reference's exact helpers so the IR / variants stay numerically
# faithful to the oracle (wllm/apps/qwen3_omni/reference/worker.py).
from wllm.apps.qwen3_omni.reference.worker import (
    ThinkerOutput,
    _build_code2wav_prompt,
    _coalesce,
    _extract_audio_chunks,
    _format_chat_prompt,
    _iter_completion_outputs,
    _take_first_marker,
    _DIST_ENV_KEYS,
)

logger = init_logger(__name__)

# Mirror wllm/apps/qwen3_omni/reference/worker.py::_init_engines: the Code2Wav
# stage config sets max_model_len=131072 (> the model's 65536 position
# limit) so the engine subprocess needs this flag, and vLLM-Omni spawns
# child workers per engine.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")


@contextmanager
def removed_envs(*names: str):
    old = os.environ.copy()
    try:
        for name in names:
            os.environ.pop(name, None)
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


@contextmanager
def pinned_visible_devices(value):
    """Temporarily set CUDA_VISIBLE_DEVICES for spawning a child AsyncOmni
    engine, restoring afterward (mirrors the reference worker._init_engines
    juggling). ``value`` may be a single physical index (int) or a CSV
    string of physical indices (e.g. "1,2,3,4" for tensor parallelism)."""
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        if value is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(value) if not isinstance(value, int) else str(int(value))
        yield
    finally:
        if prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev


# ----------------------------------------------------------------------
# Thinker
# ----------------------------------------------------------------------

def make_thinker_engine(cfg: Qwen3OmniReferenceConfig,
                        visible_devices=None,
                        stage_configs_path: Optional[str] = None) -> AsyncOmni:
    # "INHERIT" => do NOT pin CVD; the engine inherits the worker's CVD and
    # places itself via its stage-config `devices`. Used by TP variants so
    # CVD never changes in the parent (which would corrupt device_count and
    # the in-process talker's device). None => default single-GPU pin.
    if visible_devices == "INHERIT":
        cvd = None
    else:
        cvd = visible_devices if visible_devices is not None else cfg.thinker.gpu_index
    scp = stage_configs_path or cfg.thinker.stage_configs_path
    with pinned_visible_devices(cvd):
        with removed_envs(*_DIST_ENV_KEYS):
            engine = AsyncOmni(
                model=cfg.thinker.model_path,
                stage_configs_path=scp,
                trust_remote_code=True,
                **cfg.thinker.extra_engine_kwargs,
            )
    return engine


def thinker_sampling_params(cfg: Qwen3OmniReferenceConfig) -> SamplingParams:
    return SamplingParams(
        max_tokens=cfg.sampling.thinker_max_tokens,
        temperature=cfg.sampling.thinker_temperature,
        top_p=cfg.sampling.thinker_top_p,
        top_k=cfg.sampling.thinker_top_k,
        repetition_penalty=cfg.sampling.thinker_repetition_penalty,
        seed=cfg.seed,
        detokenize=True,
    )


async def _thinker_collect(engine, cfg, user_text, request_id):
    """Mirror worker._run_thinker_to_completion (drain to completion)."""
    prompt = {"prompt": _format_chat_prompt(cfg.system_prompt, user_text)}
    sp = thinker_sampling_params(cfg)
    final_out = None
    yields = 0
    async for out in engine.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=[sp],
        output_modalities=["text"],
    ):
        yields += 1
        final_out = out
    if final_out is None:
        raise RuntimeError("Thinker produced no output yields")
    return _thinker_output_from(final_out, yields)


def _thinker_output_from(final_out, yields: int) -> ThinkerOutput:
    completion = next(iter(_iter_completion_outputs(final_out)), None)
    if completion is None:
        raise RuntimeError("Thinker yielded no completion outputs")
    mm = getattr(completion, "multimodal_output", None) or {}
    embed_table = _coalesce(mm.get("0"))
    hidden_table = _coalesce(mm.get("24"))
    if embed_table is None or hidden_table is None:
        raise RuntimeError("Thinker output missing layer-0/final-layer tables")
    request_output = getattr(final_out, "request_output", None)
    prompt_token_ids = list(getattr(request_output, "prompt_token_ids", []) or [])
    output_token_ids = list(getattr(completion, "token_ids", []) or [])
    text_so_far = getattr(completion, "text", "") or ""
    if not prompt_token_ids or not output_token_ids:
        raise RuntimeError("Thinker output missing prompt/decode token ids")
    tts_bos = _take_first_marker(mm.get("tts_bos_embed"))
    tts_eos = _take_first_marker(mm.get("tts_eos_embed"))
    tts_pad = _take_first_marker(mm.get("tts_pad_embed"))
    if tts_bos is None or tts_eos is None or tts_pad is None:
        raise RuntimeError("Thinker output missing tts markers")
    return ThinkerOutput(
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_token_ids,
        embed_table=embed_table,
        hidden_table=hidden_table,
        tts_bos_embed=tts_bos,
        tts_eos_embed=tts_eos,
        tts_pad_embed=tts_pad,
        text=text_so_far,
    )


def run_thinker(engine, cfg, user_text: str, request_id: str,
                runner: Optional[asyncio.Runner] = None) -> ThinkerOutput:
    if runner is not None:
        return runner.run(_thinker_collect(engine, cfg, user_text, request_id))
    return asyncio.run(_thinker_collect(engine, cfg, user_text, request_id))


# ----------------------------------------------------------------------
# Talker
# ----------------------------------------------------------------------

def make_talker_runner(cfg: Qwen3OmniReferenceConfig) -> Qwen3OmniTalkerRunner:
    return Qwen3OmniTalkerRunner(
        cfg.talker.model_path,
        gpu_index=int(cfg.talker.gpu_index) if cfg.talker.gpu_index is not None else 0,
        temperature=cfg.sampling.talker_temperature,
        top_k=cfg.sampling.talker_top_k,
        top_p=cfg.sampling.talker_top_p,
        repetition_penalty=cfg.sampling.talker_repetition_penalty,
        seed=cfg.seed,
        max_tokens=cfg.sampling.talker_max_tokens,
        max_seq_len=cfg.sampling.talker_max_seq_len,
    )


def prime_talker(runner: Qwen3OmniTalkerRunner, thinker: ThinkerOutput,
                 cfg: Qwen3OmniReferenceConfig, *, push_all: bool = True) -> None:
    """Reset + prefill the talker from a ThinkerOutput.

    Mirrors worker._run_talker_to_completion's priming. When push_all is
    True (reference behavior) all remaining thinker decode embeds are
    pushed and the thinker session is marked finished immediately. When
    False, the caller is responsible for streaming them in (variant use).
    """
    runner._reset_session_state()
    prefill_len = len(thinker.prompt_token_ids)
    embed_rows = int(thinker.embed_table.shape[0])
    runner.start_session(
        thinker_prompt_token_ids=thinker.prompt_token_ids,
        thinker_output_token_ids=[thinker.output_token_ids[0]],
        thinker_prefill_embed=thinker.embed_table[: prefill_len + 1],
        thinker_prefill_hidden=thinker.hidden_table[: prefill_len + 1],
        tts_bos_embed_thinker=thinker.tts_bos_embed,
        tts_eos_embed_thinker=thinker.tts_eos_embed,
        tts_pad_embed_thinker=thinker.tts_pad_embed,
        speaker=cfg.speaker,
    )
    if push_all:
        remaining_start = prefill_len + 1
        available = min(embed_rows - prefill_len,
                        max(len(thinker.output_token_ids) - 1, 0))
        remaining_end = prefill_len + available
        if remaining_end > remaining_start:
            runner.append_thinker_decode_token(
                thinker.embed_table[remaining_start:remaining_end])
        runner.mark_thinker_finished()


def talker_remaining_embeds(thinker: ThinkerOutput) -> torch.Tensor:
    """The decode-token embeds that the reference bulk-pushes after prefill.

    Row i corresponds to thinker decode token (prefill_len + 1 + i). Used
    by streaming variants to push them in incrementally.
    """
    prefill_len = len(thinker.prompt_token_ids)
    embed_rows = int(thinker.embed_table.shape[0])
    remaining_start = prefill_len + 1
    available = min(embed_rows - prefill_len,
                    max(len(thinker.output_token_ids) - 1, 0))
    remaining_end = prefill_len + available
    if remaining_end > remaining_start:
        return thinker.embed_table[remaining_start:remaining_end]
    return thinker.embed_table[0:0]


def run_talker_to_completion(runner: Qwen3OmniTalkerRunner) -> List[torch.Tensor]:
    """Step the (already primed) runner until codec EOS. Reference path."""
    frames: List[torch.Tensor] = []
    while True:
        frame = runner.step()
        if frame is None:
            if not runner.is_done():
                raise RuntimeError("Talker step()=None but not done; queue underrun")
            break
        frames.append(frame)
    return frames


# ----------------------------------------------------------------------
# Code2Wav
# ----------------------------------------------------------------------

def make_c2w_engine(cfg: Qwen3OmniReferenceConfig,
                    visible_devices=None,
                    stage_configs_path: Optional[str] = None) -> AsyncOmni:
    if visible_devices == "INHERIT":
        cvd = None
    else:
        cvd = visible_devices if visible_devices is not None else cfg.code2wav.gpu_index
    scp = stage_configs_path or cfg.code2wav.stage_configs_path
    with pinned_visible_devices(cvd):
        with removed_envs(*_DIST_ENV_KEYS):
            engine = AsyncOmni(
                model=cfg.code2wav.model_path,
                stage_configs_path=scp,
                trust_remote_code=True,
                **cfg.code2wav.extra_engine_kwargs,
            )
    return engine


def c2w_sampling_params(cfg: Qwen3OmniReferenceConfig) -> SamplingParams:
    return SamplingParams(
        max_tokens=cfg.sampling.code2wav_max_tokens,
        temperature=cfg.sampling.code2wav_temperature,
        top_p=cfg.sampling.code2wav_top_p,
        repetition_penalty=cfg.sampling.code2wav_repetition_penalty,
        seed=cfg.seed,
        detokenize=True,
    )


async def _c2w_collect(engine, cfg, frames, request_id):
    """Mirror worker._run_code2wav_to_completion (single full request)."""
    if not frames:
        return np.empty(0, dtype=np.float32), int(cfg.audio_sample_rate)
    chunk_codes = torch.stack(frames, dim=0)
    prompt = _build_code2wav_prompt(chunk_codes)
    sp = c2w_sampling_params(cfg)
    consumed = 0
    sample_rate = int(cfg.audio_sample_rate)
    chunks: List[np.ndarray] = []
    async for omni_out in engine.generate(
        prompt=prompt,
        request_id=request_id,
        sampling_params_list=[sp],
        output_modalities=["audio"],
    ):
        new_chunks, consumed, sr = _extract_audio_chunks(omni_out, consumed)
        if sr is not None:
            sample_rate = sr
        for chunk in new_chunks:
            if not isinstance(chunk, torch.Tensor):
                chunk = torch.as_tensor(chunk)
            np_chunk = chunk.float().detach().cpu().numpy().reshape(-1)
            if np_chunk.size > 0:
                chunks.append(np_chunk)
    audio = (np.concatenate(chunks).astype(np.float32, copy=False)
             if chunks else np.empty(0, dtype=np.float32))
    return audio, sample_rate


def vocode_full(engine, cfg, frames: List[torch.Tensor], request_id: str,
                runner: Optional[asyncio.Runner] = None) -> Tuple[np.ndarray, int]:
    if runner is not None:
        return runner.run(_c2w_collect(engine, cfg, frames, request_id))
    return asyncio.run(_c2w_collect(engine, cfg, frames, request_id))
