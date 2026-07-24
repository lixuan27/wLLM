"""Reusable Qwen3-Omni pipeline components.

Factors the reference worker's three stages into reusable pieces that
BOTH the IR conversion (Phase 1-2) and the deployment variants (Phase
3-4) build on. The low-level helper functions and the ``ThinkerOutput``
dataclass are copied verbatim from the reference worker
(``wllm/apps/qwen3_omni/reference/worker.py``) because they define the exact
numerical contract we must preserve.

Three components:
  * ``ThinkerComponent``  — wraps an AsyncOmni thinker engine. Supports
    both whole-response extraction (reference-faithful) and incremental
    streaming of decode-token embeddings (for pipelined variants).
  * ``TalkerComponent``   — wraps the (vendored) Qwen3OmniTalkerRunner.
  * ``Code2WavComponent`` — wraps an AsyncOmni code2wav engine. Supports
    whole-sequence vocoding (reference) and chunked streaming vocoding
    with left context (native Qwen3-Omni async_chunk design).

Device placement is a parameter, not hard-coded — every component takes
its own physical GPU index (and, for streaming variants, engines run in
their own subprocess-managed AsyncOmni which pins via CUDA_VISIBLE_DEVICES).
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import torch

from vllm_omni import AsyncOmni
from vllm.sampling_params import SamplingParams

from wllm.serving.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# distributed env scrubbing (verbatim from reference)
# ---------------------------------------------------------------------------

_DIST_ENV_KEYS = (
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK",
    "ROLE_RANK", "ROLE_NAME", "OMP_NUM_THREADS", "MASTER_ADDR", "MASTER_PORT",
    "TORCHELASTIC_USE_AGENT_STORE", "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID", "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCHELASTIC_ERROR_FILE",
)


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
def visible_devices(devices: Optional[str]):
    """Temporarily set CUDA_VISIBLE_DEVICES (physical ids) then restore."""
    prev = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        if devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
        yield
    finally:
        if prev is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev


# ---------------------------------------------------------------------------
# helpers (verbatim from reference worker — the numerical contract)
# ---------------------------------------------------------------------------


def format_chat_prompt(system_prompt: str, user_text: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def iter_completion_outputs(omni_output) -> Iterable[Any]:
    request_output = getattr(omni_output, "request_output", None)
    outputs = getattr(request_output, "outputs", None) if request_output else None
    return outputs or ()


def coalesce(t):
    if t is None:
        return None
    if isinstance(t, list):
        if not t:
            return None
        t = torch.cat(
            [x if isinstance(x, torch.Tensor) else torch.as_tensor(x) for x in t],
            dim=0,
        )
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return None
    return t


def take_first_marker(t):
    if t is None:
        return None
    if isinstance(t, list):
        if not t:
            return None
        t = t[0]
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    while t.ndim > 1 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim == 1:
        return t.unsqueeze(0)
    if t.ndim == 2:
        return t[:1]
    return t.reshape(-1, t.shape[-1])[:1]


def build_code2wav_prompt(codec_codes: torch.Tensor) -> dict:
    if codec_codes.ndim != 2:
        raise ValueError(f"expected 2-D codec codes, got shape {tuple(codec_codes.shape)}")
    if codec_codes.shape[0] != 16 and codec_codes.shape[1] == 16:
        codec_codes = codec_codes.transpose(0, 1).contiguous()
    flat = codec_codes.to(torch.long).reshape(-1).tolist()
    return {"prompt_token_ids": flat}


def extract_audio_chunks(omni_output, already_consumed: int
                         ) -> Tuple[List[torch.Tensor], int, Optional[int]]:
    completion = next(iter(iter_completion_outputs(omni_output)), None)
    if completion is None:
        return [], already_consumed, None
    mm = getattr(completion, "multimodal_output", None) or {}
    audio = mm.get("audio")
    sr = mm.get("sr")
    if isinstance(sr, torch.Tensor):
        sr = int(sr.item())
    elif sr is not None:
        sr = int(sr)
    if isinstance(audio, list):
        new = audio[already_consumed:]
        return new, len(audio), sr
    if isinstance(audio, torch.Tensor):
        return [audio], already_consumed + 1, sr
    return [], already_consumed, sr


@dataclass
class ThinkerOutput:
    """The full set of values the Talker needs from a finished Thinker run."""
    prompt_token_ids: List[int]
    output_token_ids: List[int]
    embed_table: torch.Tensor    # [P + N - 1, hidden_thinker]
    hidden_table: torch.Tensor   # [P + N - 1, hidden_thinker]
    tts_bos_embed: torch.Tensor  # [1, hidden_thinker]
    tts_eos_embed: torch.Tensor
    tts_pad_embed: torch.Tensor
    text: str


# ---------------------------------------------------------------------------
# component configs
# ---------------------------------------------------------------------------


@dataclass
class ThinkerCfg:
    model_path: str
    stage_configs_path: str
    gpu_index: int = 0
    tensor_parallel_size: int = 1
    extra_engine_kwargs: dict = None
    max_tokens: int = 2048
    temperature: float = 0.4
    top_p: float = 0.9
    top_k: int = 1
    repetition_penalty: float = 1.05
    seed: int = 42


@dataclass
class TalkerCfg:
    model_path: str
    gpu_index: int = 1
    tensor_parallel_size: int = 1
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    max_tokens: int = 4096
    max_seq_len: int = 8192
    seed: int = 42


@dataclass
class Code2WavCfg:
    model_path: str
    stage_configs_path: str
    gpu_index: int = 1
    tensor_parallel_size: int = 1
    extra_engine_kwargs: dict = None
    max_tokens: int = 65536
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.1
    seed: int = 42


# ---------------------------------------------------------------------------
# Thinker component
# ---------------------------------------------------------------------------


class ThinkerComponent:
    """AsyncOmni thinker engine wrapper. Pins to ``cfg.gpu_index`` (physical)."""

    def __init__(self, cfg: ThinkerCfg, system_prompt: str):
        self.cfg = cfg
        self.system_prompt = system_prompt
        extra = dict(cfg.extra_engine_kwargs or {})
        if cfg.tensor_parallel_size > 1:
            extra.setdefault("tensor_parallel_size", cfg.tensor_parallel_size)
        with visible_devices(self._device_str()), removed_envs(*_DIST_ENV_KEYS):
            logger.info("Loading Thinker from %s on physical GPU(s) %s",
                        cfg.model_path, self._device_str())
            self.engine = AsyncOmni(
                model=cfg.model_path,
                stage_configs_path=cfg.stage_configs_path,
                trust_remote_code=True, **extra,
            )

    def _device_str(self) -> str:
        n = self.cfg.tensor_parallel_size
        base = int(self.cfg.gpu_index)
        return ",".join(str(base + i) for i in range(n))

    def _sampling(self) -> SamplingParams:
        c = self.cfg
        return SamplingParams(
            max_tokens=c.max_tokens, temperature=c.temperature, top_p=c.top_p,
            top_k=c.top_k, repetition_penalty=c.repetition_penalty, seed=c.seed,
            detokenize=True,
        )

    async def run_to_completion(self, user_text: str, request_id: str) -> ThinkerOutput:
        """Reference-faithful: drain the thinker, return full decode state."""
        prompt = {"prompt": format_chat_prompt(self.system_prompt, user_text)}
        final_out = None
        async for out in self.engine.generate(
            prompt=prompt, request_id=request_id,
            sampling_params_list=[self._sampling()], output_modalities=["text"],
        ):
            final_out = out
        return self._finalize(final_out)

    async def stream(self, user_text: str, request_id: str):
        """Yield (embed_table, hidden_table, markers_or_None, finished_output_or_None)
        deltas as the thinker decodes. On the final yield returns a full
        ThinkerOutput. Consumers prime the talker once enough rows exist and
        append new rows as they arrive.

        Yields tuples: (embed_table, hidden_table, prompt_token_ids,
                        output_token_ids, markers, is_final, thinker_output).
        markers = (tts_bos, tts_eos, tts_pad) once available, else None.
        """
        prompt = {"prompt": format_chat_prompt(self.system_prompt, user_text)}
        final_out = None
        async for out in self.engine.generate(
            prompt=prompt, request_id=request_id,
            sampling_params_list=[self._sampling()], output_modalities=["text"],
        ):
            final_out = out
            completion = next(iter(iter_completion_outputs(out)), None)
            if completion is None:
                continue
            mm = getattr(completion, "multimodal_output", None) or {}
            embed = coalesce(mm.get("0"))
            hidden = coalesce(mm.get("24"))
            if embed is None or hidden is None:
                continue
            ro = getattr(out, "request_output", None)
            ptids = list(getattr(ro, "prompt_token_ids", []) or [])
            otids = list(getattr(completion, "token_ids", []) or [])
            bos = take_first_marker(mm.get("tts_bos_embed"))
            eos = take_first_marker(mm.get("tts_eos_embed"))
            pad = take_first_marker(mm.get("tts_pad_embed"))
            markers = (bos, eos, pad) if (bos is not None and eos is not None and pad is not None) else None
            yield (embed, hidden, ptids, otids, markers, False, None)
        # final
        to = self._finalize(final_out)
        yield (to.embed_table, to.hidden_table, to.prompt_token_ids,
               to.output_token_ids, (to.tts_bos_embed, to.tts_eos_embed, to.tts_pad_embed),
               True, to)

    def _finalize(self, final_out) -> ThinkerOutput:
        if final_out is None:
            raise RuntimeError("Thinker engine produced no output yields")
        completion = next(iter(iter_completion_outputs(final_out)), None)
        if completion is None:
            raise RuntimeError("Thinker yielded no completion outputs")
        mm = getattr(completion, "multimodal_output", None) or {}
        embed_table = coalesce(mm.get("0"))
        hidden_table = coalesce(mm.get("24"))
        if embed_table is None or hidden_table is None:
            raise RuntimeError("Thinker output missing layer-0 / final-layer tables")
        ro = getattr(final_out, "request_output", None)
        prompt_token_ids = list(getattr(ro, "prompt_token_ids", []) or [])
        output_token_ids = list(getattr(completion, "token_ids", []) or [])
        text_so_far = getattr(completion, "text", "") or ""
        if not prompt_token_ids:
            raise RuntimeError("Thinker output has no prompt token ids")
        if not output_token_ids:
            raise RuntimeError("Thinker output has no decode tokens")
        bos = take_first_marker(mm.get("tts_bos_embed"))
        eos = take_first_marker(mm.get("tts_eos_embed"))
        pad = take_first_marker(mm.get("tts_pad_embed"))
        if bos is None or eos is None or pad is None:
            raise RuntimeError("Thinker output missing tts markers")
        return ThinkerOutput(
            prompt_token_ids=prompt_token_ids, output_token_ids=output_token_ids,
            embed_table=embed_table, hidden_table=hidden_table,
            tts_bos_embed=bos, tts_eos_embed=eos, tts_pad_embed=pad, text=text_so_far,
        )

    def shutdown(self):
        try:
            self.engine.shutdown()
        except Exception:
            logger.exception("Error shutting down thinker engine")


# ---------------------------------------------------------------------------
# Talker component (wraps the vendored runner)
# ---------------------------------------------------------------------------


class TalkerComponent:
    """Wraps the vendored Qwen3OmniTalkerRunner. Reference-faithful priming +
    per-step generation, plus incremental token appends for streaming."""

    def __init__(self, cfg: TalkerCfg, speaker: str, runner_cls=None):
        self.cfg = cfg
        self.speaker = speaker
        if runner_cls is None:
            from wllm.apps.qwen3_omni.backend.rocm.pipeline.talker_runner import Qwen3OmniTalkerRunner
            runner_cls = Qwen3OmniTalkerRunner
        self.runner = runner_cls(
            cfg.model_path, gpu_index=int(cfg.gpu_index),
            temperature=cfg.temperature, top_k=cfg.top_k, top_p=cfg.top_p,
            repetition_penalty=cfg.repetition_penalty, seed=cfg.seed,
            max_tokens=cfg.max_tokens, max_seq_len=cfg.max_seq_len,
        )

    def prime_whole(self, thinker: ThinkerOutput) -> None:
        """Reference-faithful priming: prefill + bulk-append all trailing
        decode embeds + mark finished (matches worker._run_talker_to_completion)."""
        r = self.runner
        r._reset_session_state()
        prefill_len = len(thinker.prompt_token_ids)
        embed_rows = int(thinker.embed_table.shape[0])
        r.start_session(
            thinker_prompt_token_ids=thinker.prompt_token_ids,
            thinker_output_token_ids=[thinker.output_token_ids[0]],
            thinker_prefill_embed=thinker.embed_table[: prefill_len + 1],
            thinker_prefill_hidden=thinker.hidden_table[: prefill_len + 1],
            tts_bos_embed_thinker=thinker.tts_bos_embed,
            tts_eos_embed_thinker=thinker.tts_eos_embed,
            tts_pad_embed_thinker=thinker.tts_pad_embed,
            speaker=self.speaker,
        )
        remaining_start = prefill_len + 1
        available_embeds = min(embed_rows - prefill_len,
                               max(len(thinker.output_token_ids) - 1, 0))
        remaining_end = prefill_len + available_embeds
        if remaining_end > remaining_start:
            r.append_thinker_decode_token(thinker.embed_table[remaining_start:remaining_end])
        r.mark_thinker_finished()

    def run_to_completion(self, thinker: ThinkerOutput) -> List[torch.Tensor]:
        self.prime_whole(thinker)
        frames: List[torch.Tensor] = []
        while True:
            frame = self.runner.step()
            if frame is None:
                if not self.runner.is_done():
                    raise RuntimeError("talker step None but not done (queue underrun?)")
                break
            frames.append(frame)
        return frames

    def shutdown(self):
        try:
            self.runner.shutdown()
        except Exception:
            logger.exception("Error shutting down talker runner")


# ---------------------------------------------------------------------------
# Code2Wav component
# ---------------------------------------------------------------------------


class Code2WavComponent:
    """AsyncOmni code2wav engine wrapper. Whole-sequence (reference) and
    chunked streaming (native async_chunk) vocoding."""

    def __init__(self, cfg: Code2WavCfg):
        self.cfg = cfg
        extra = dict(cfg.extra_engine_kwargs or {})
        if cfg.tensor_parallel_size > 1:
            extra.setdefault("tensor_parallel_size", cfg.tensor_parallel_size)
        with visible_devices(self._device_str()), removed_envs(*_DIST_ENV_KEYS):
            logger.info("Loading Code2Wav from %s on physical GPU(s) %s",
                        cfg.model_path, self._device_str())
            self.engine = AsyncOmni(
                model=cfg.model_path,
                stage_configs_path=cfg.stage_configs_path,
                trust_remote_code=True, **extra,
            )

    def _device_str(self) -> str:
        n = self.cfg.tensor_parallel_size
        base = int(self.cfg.gpu_index)
        return ",".join(str(base + i) for i in range(n))

    def _sampling(self) -> SamplingParams:
        c = self.cfg
        return SamplingParams(
            max_tokens=c.max_tokens, temperature=c.temperature, top_p=c.top_p,
            repetition_penalty=c.repetition_penalty, seed=c.seed, detokenize=True,
        )

    async def warmup(self, n_frames: int = 8, request_id: str = "warmup-c2w") -> None:
        dummy = torch.zeros(n_frames, 16, dtype=torch.long)
        prompt = build_code2wav_prompt(dummy)
        async for _ in self.engine.generate(
            prompt=prompt, request_id=request_id,
            sampling_params_list=[self._sampling()], output_modalities=["audio"],
        ):
            pass

    async def vocode(self, codec_frames: List[torch.Tensor], request_id: str,
                     default_sr: int) -> Tuple[np.ndarray, int]:
        """Whole-sequence vocoding (reference-faithful). One request over all
        frames, concatenate every audio chunk yielded."""
        if not codec_frames:
            return np.empty(0, dtype=np.float32), int(default_sr)
        chunk_codes = torch.stack(codec_frames, dim=0)
        prompt = build_code2wav_prompt(chunk_codes)
        consumed = 0
        sample_rate = int(default_sr)
        chunks: List[np.ndarray] = []
        async for omni_out in self.engine.generate(
            prompt=prompt, request_id=request_id,
            sampling_params_list=[self._sampling()], output_modalities=["audio"],
        ):
            new_chunks, consumed, sr = extract_audio_chunks(omni_out, consumed)
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

    async def vocode_chunk(self, chunk_with_context: List[torch.Tensor],
                           n_new_frames: int, samples_per_frame: int,
                           request_id: str, default_sr: int) -> Tuple[np.ndarray, int]:
        """Chunked streaming vocoding with left context. Vocode the whole
        [left_context + new] window, then emit only the trailing
        ``n_new_frames * samples_per_frame`` samples (left-context audio was
        already emitted by the previous chunk). Matches the native
        async_chunk connector's codec_left_context_frames handling."""
        audio, sr = await self.vocode(chunk_with_context, request_id, default_sr)
        # The new frames are at the END of the window, so their audio is the
        # trailing portion. Emit the last n_new_frames*spf samples — robust to
        # vocoder edge padding (output length may not be exactly frames*spf).
        n_ctx_frames = len(chunk_with_context) - n_new_frames
        if n_ctx_frames > 0:
            want = n_new_frames * samples_per_frame
            audio = audio[-want:] if audio.size > want else audio
        return audio.astype(np.float32, copy=False), sr

    def shutdown(self):
        try:
            self.engine.shutdown()
        except Exception:
            logger.exception("Error shutting down code2wav engine")
