"""Reference single-process sequential Qwen3-Omni worker.

The Qwen3-Omni response pipeline is

    text_prompt
        -> Thinker     (LLM, generates the text + per-token embeddings)
        -> Talker      (LLM over the thinker output, generates RVQ codec frames)
        -> Code2Wav    (decoder, turns codec frames into raw audio)
        -> audio_output_buffer

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

from wllm.apps.qwen3_omni.adapter import (
    TEXT_FRAME_BYTES,
    decode_text_frame,
)
from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.logger import init_logger
from wllm.apps.qwen3_omni.reference.config import Qwen3OmniReferenceConfig
from wllm.apps.qwen3_omni.reference.runner import Qwen3OmniTalkerRunner
from wllm.serving.utils.rand import set_global_seed
from wllm.serving.utils.torch_utils import set_torch_options

logger = init_logger(__name__)
set_torch_options()


# Distributed env vars to scrub when bringing up an AsyncOmni engine
# from inside a torchrun-launched parent (matches LiveAvatar's list).
_DIST_ENV_KEYS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_NAME",
    "OMP_NUM_THREADS",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_USE_AGENT_STORE",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCHELASTIC_ERROR_FILE",
)


@contextmanager
def _removed_envs(*names: str):
    """Temporarily drop the named env vars; restore on exit."""
    old = os.environ.copy()
    try:
        for name in names:
            os.environ.pop(name, None)
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_chat_prompt(system_prompt: str, user_text: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _iter_completion_outputs(omni_output) -> Iterable[Any]:
    request_output = getattr(omni_output, "request_output", None)
    outputs = getattr(request_output, "outputs", None) if request_output else None
    return outputs or ()


def _coalesce(t):
    """multimodal_output entries arrive as either a tensor or a list of
    [N_step, hidden] tensors that need concatenating along the row
    axis. Reduce to a single 2-D tensor or return None if empty."""
    if t is None:
        return None
    if isinstance(t, list):
        if not t:
            return None
        t = torch.cat(
            [
                x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
                for x in t
            ],
            dim=0,
        )
    if not isinstance(t, torch.Tensor) or t.numel() == 0:
        return None
    return t


def _take_first_marker(t):
    """Reduce a TTS-marker entry (tts_bos_embed / tts_eos_embed /
    tts_pad_embed) to a single row of shape ``[1, hidden_thinker]``.
    The thinker emits the marker as ``[1, 1, hidden]`` per forward step
    and the output processor accumulates per-step copies in a list, but
    the marker is constant so we just take the first element."""
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


def _build_code2wav_prompt(codec_codes: torch.Tensor) -> dict:
    """Flatten ``[T, 16]`` RVQ codes into Code2Wav's prompt token ids.

    The model reshapes ``input_ids`` as ``[1, 16, T]`` (layer-major), so
    we transpose ``[T, 16]`` -> ``[16, T]`` before flattening.
    """
    if codec_codes.ndim != 2:
        raise ValueError(
            f"expected 2-D codec codes, got shape {tuple(codec_codes.shape)}"
        )
    if codec_codes.shape[0] != 16 and codec_codes.shape[1] == 16:
        codec_codes = codec_codes.transpose(0, 1).contiguous()
    flat = codec_codes.to(torch.long).reshape(-1).tolist()
    return {"prompt_token_ids": flat}


def _extract_audio_chunks(
    omni_output, already_consumed: int,
) -> Tuple[List[torch.Tensor], int, Optional[int]]:
    completion = next(iter(_iter_completion_outputs(omni_output)), None)
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


# ---------------------------------------------------------------------------
# Thinker output container
# ---------------------------------------------------------------------------


@dataclass
class ThinkerOutput:
    """The full set of values the Talker needs from a finished Thinker run."""

    prompt_token_ids: List[int]
    output_token_ids: List[int]
    embed_table: torch.Tensor   # [P + N - 1, hidden_thinker] (last sampled
                                # token's embed isn't computed until the
                                # next decode forward and is therefore
                                # missing from the final yield).
    hidden_table: torch.Tensor  # [P + N - 1, hidden_thinker]
    tts_bos_embed: torch.Tensor   # [1, hidden_thinker]
    tts_eos_embed: torch.Tensor   # [1, hidden_thinker]
    tts_pad_embed: torch.Tensor   # [1, hidden_thinker]
    text: str


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class Qwen3OmniWorker:
    """Sequential single-process reference worker for Qwen3-Omni."""

    # Control commands (match the adapter / other reference workers).
    _CMD_START = 1
    _CMD_TERMINATE = 2
    _CMD_RESET = 3

    def __init__(self, cfg_path: str):
        self.reference_cfg = Qwen3OmniReferenceConfig.from_yaml(cfg_path, is_path=True)
        self.cfg = self.reference_cfg
        self._init_worker()
        logger.info("Qwen3-Omni backend READY")

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _init_worker(self):
        set_global_seed(self.cfg.seed)

        # Channels are owned by this worker (single-process).
        self.ctrl_buffer = SharedControlBuffer(
            self.cfg.ctrl_buffer_name, create=True,
        )
        self.text_input_buffer = SharedTensorBuffer(
            name=self.cfg.text_input_buffer_name,
            frame_shape=(TEXT_FRAME_BYTES,),
            max_len=int(self.cfg.text_max_pending),
            dtype=np.uint8,
            create=True,
        )
        self.audio_output_buffer = SharedTensorBuffer(
            self.cfg.audio_output_buffer_name,
            frame_shape=(int(self.cfg.audio_frame_samples),),
            dtype=np.float32,
            max_len=int(self.cfg.audio_max_chunks),
            create=True,
        )
        self.audio_meta_buffer = SharedControlBuffer(
            self.cfg.audio_meta_buffer_name, create=True,
        )

        self._init_engines()

        self._async_runner = asyncio.Runner()
        self.session_started = False
        self._next_text_index = 0

        # Publish the default sample rate so the adapter has a value to
        # read before the first response arrives.
        self.audio_meta_buffer.send(
            int(self.cfg.audio_sample_rate), timeout_s=10.0,
        )

        # Warm up Code2Wav via the persistent async runner so we don't
        # close any loop the engine's resources depend on. The first
        # c2w request after engine init pays ~2-3 s of lazy one-time
        # setup; submit a dummy tiny request now so the user's first
        # prompt skips that cost.
        try:
            self._async_runner.run(self._warmup_code2wav())
        except Exception:
            logger.exception(
                "Code2Wav warmup failed; first user prompt will pay "
                "the c2w lazy-init cost"
            )

    def _init_engines(self):
        """Bring up Thinker, Talker, Code2Wav engines (eager, in order)."""
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        try:
            if self.cfg.thinker.gpu_index is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(int(self.cfg.thinker.gpu_index))
            logger.info(
                "Loading Qwen3-Omni Thinker engine from %s",
                self.cfg.thinker.model_path,
            )
            with _removed_envs(*_DIST_ENV_KEYS):
                self.thinker_engine = AsyncOmni(
                    model=self.cfg.thinker.model_path,
                    stage_configs_path=self.cfg.thinker.stage_configs_path,
                    trust_remote_code=True,
                    **self.cfg.thinker.extra_engine_kwargs,
                )

            # The Talker runs in-process; restore the parent's full
            # CUDA_VISIBLE_DEVICES so ``talker.gpu_index`` translates to
            # a valid ``cuda:N``.
            if visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

            logger.info(
                "Loading Qwen3-Omni Talker from %s",
                self.cfg.talker.model_path,
            )
            self.talker_runner = Qwen3OmniTalkerRunner(
                self.cfg.talker.model_path,
                gpu_index=int(self.cfg.talker.gpu_index)
                if self.cfg.talker.gpu_index is not None
                else 0,
                temperature=self.cfg.sampling.talker_temperature,
                top_k=self.cfg.sampling.talker_top_k,
                top_p=self.cfg.sampling.talker_top_p,
                repetition_penalty=self.cfg.sampling.talker_repetition_penalty,
                seed=self.cfg.seed,
                max_tokens=self.cfg.sampling.talker_max_tokens,
                max_seq_len=self.cfg.sampling.talker_max_seq_len,
            )

            if self.cfg.code2wav.gpu_index is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(int(self.cfg.code2wav.gpu_index))
            logger.info(
                "Loading Qwen3-Omni Code2Wav engine from %s",
                self.cfg.code2wav.model_path,
            )
            with _removed_envs(*_DIST_ENV_KEYS):
                self.code2wav_engine = AsyncOmni(
                    model=self.cfg.code2wav.model_path,
                    stage_configs_path=self.cfg.code2wav.stage_configs_path,
                    trust_remote_code=True,
                    **self.cfg.code2wav.extra_engine_kwargs,
                )
        finally:
            if visible_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    async def _warmup_code2wav(self) -> None:
        """Submit a tiny dummy c2w request so the engine's one-time lazy
        init is paid during backend startup rather than on the first user
        prompt."""
        warmup_t0 = time.time()
        n_frames = 8
        dummy_codes = torch.zeros(n_frames, 16, dtype=torch.long)

        prompt = _build_code2wav_prompt(dummy_codes)
        sp = SamplingParams(
            max_tokens=self.cfg.sampling.code2wav_max_tokens,
            temperature=self.cfg.sampling.code2wav_temperature,
            top_p=self.cfg.sampling.code2wav_top_p,
            repetition_penalty=self.cfg.sampling.code2wav_repetition_penalty,
            seed=self.cfg.seed,
            detokenize=True,
        )
        async for _ in self.code2wav_engine.generate(
            prompt=prompt,
            request_id="warmup-c2w-0",
            sampling_params_list=[sp],
            output_modalities=["audio"],
        ):
            pass

        logger.info(
            "Code2Wav warmup complete in %.2fs (dummy %d-frame chunk)",
            time.time() - warmup_t0, n_frames,
        )

    # ------------------------------------------------------------------
    # session lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self.session_started = True
        self.audio_output_buffer.clear()
        self._next_text_index = self.text_input_buffer.num
        self.ctrl_buffer.commit()
        logger.info("Qwen3-Omni session started")

    def reset(self):
        self.session_started = False
        self.audio_output_buffer.clear()
        self._next_text_index = self.text_input_buffer.num
        self.ctrl_buffer.commit()
        logger.info("Qwen3-Omni session reset")

    def terminate(self):
        self.session_started = False
        for name, engine in (
            ("thinker", getattr(self, "thinker_engine", None)),
            ("code2wav", getattr(self, "code2wav_engine", None)),
        ):
            if engine is None:
                continue
            try:
                engine.shutdown()
            except Exception:
                logger.exception("Error shutting down %s engine", name)
        if getattr(self, "talker_runner", None) is not None:
            try:
                self.talker_runner.shutdown()
            except Exception:
                logger.exception("Error shutting down talker runner")
            self.talker_runner = None
        if getattr(self, "_async_runner", None) is not None:
            try:
                self._async_runner.close()
            except Exception:
                pass
            self._async_runner = None

        self.ctrl_buffer.commit()
        for buf, name in (
            (self.ctrl_buffer, "ctrl_buffer"),
            (self.text_input_buffer, "text_input_buffer"),
            (self.audio_output_buffer, "audio_output_buffer"),
            (self.audio_meta_buffer, "audio_meta_buffer"),
        ):
            try:
                buf.unlink()
            except Exception:
                logger.exception("Error while unlinking %s", name)
        logger.info("Qwen3-Omni worker terminated")

    # ------------------------------------------------------------------
    # one prompt: thinker -> talker -> code2wav (each to completion)
    # ------------------------------------------------------------------

    def _read_pending_text(self) -> Optional[str]:
        next_idx, frame = self.text_input_buffer.read(self._next_text_index, 1)
        if frame is None:
            self._next_text_index = next_idx
            return None
        self._next_text_index = next_idx
        return decode_text_frame(frame[0])

    async def _run_thinker_to_completion(
        self, user_text: str, request_id: str,
    ) -> ThinkerOutput:
        """Drain the thinker engine; return the full prompt + decode state.

        The reference worker does NOT feed the thinker's intermediate
        yields into the talker -- it waits until the thinker has emitted
        every decode token before starting the talker.
        """
        thinker_prompt = {"prompt": _format_chat_prompt(self.cfg.system_prompt, user_text)}
        thinker_sp = SamplingParams(
            max_tokens=self.cfg.sampling.thinker_max_tokens,
            temperature=self.cfg.sampling.thinker_temperature,
            top_p=self.cfg.sampling.thinker_top_p,
            top_k=self.cfg.sampling.thinker_top_k,
            repetition_penalty=self.cfg.sampling.thinker_repetition_penalty,
            seed=self.cfg.seed,
            detokenize=True,
        )

        final_out = None
        thinker_yields = 0
        async for out in self.thinker_engine.generate(
            prompt=thinker_prompt,
            request_id=request_id,
            sampling_params_list=[thinker_sp],
            output_modalities=["text"],
        ):
            thinker_yields += 1
            final_out = out

        if final_out is None:
            raise RuntimeError("Thinker engine produced no output yields")

        completion = next(iter(_iter_completion_outputs(final_out)), None)
        if completion is None:
            raise RuntimeError("Thinker yielded no completion outputs")

        mm = getattr(completion, "multimodal_output", None) or {}
        embed_table = _coalesce(mm.get("0"))
        hidden_table = _coalesce(mm.get("24"))
        if embed_table is None or hidden_table is None:
            raise RuntimeError(
                "Thinker output missing layer-0 / final-layer embed/hidden tables"
            )

        request_output = getattr(final_out, "request_output", None)
        prompt_token_ids = list(getattr(request_output, "prompt_token_ids", []) or [])
        output_token_ids = list(getattr(completion, "token_ids", []) or [])
        text_so_far = getattr(completion, "text", "") or ""
        if not prompt_token_ids:
            raise RuntimeError("Thinker output has no prompt token ids")
        if not output_token_ids:
            raise RuntimeError("Thinker output has no decode tokens")

        tts_bos = _take_first_marker(mm.get("tts_bos_embed"))
        tts_eos = _take_first_marker(mm.get("tts_eos_embed"))
        tts_pad = _take_first_marker(mm.get("tts_pad_embed"))
        if tts_bos is None or tts_eos is None or tts_pad is None:
            raise RuntimeError("Thinker output missing tts_bos/eos/pad markers")

        logger.info(
            "Thinker finished: %d decode tokens, %d engine yields. Sample: %s",
            len(output_token_ids),
            thinker_yields,
            text_so_far[:80].replace("\n", " "),
        )

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

    def _run_talker_to_completion(
        self, thinker: ThinkerOutput,
    ) -> List[torch.Tensor]:
        """Run the talker over the full thinker output; collect all codec frames.

        Resets the runner, prefills with prompt + first thinker token,
        bulk-pushes the remaining decode-token embeddings into the
        trailing queue, marks the thinker session finished, then loops
        ``step()`` until codec EOS is emitted.
        """
        # Wipe any previous session before priming the new one.
        self.talker_runner._reset_session_state()

        prefill_len = len(thinker.prompt_token_ids)
        embed_rows = int(thinker.embed_table.shape[0])

        # Prefill needs prompt + 1 generated token (P + 1 rows of embed/hidden).
        self.talker_runner.start_session(
            thinker_prompt_token_ids=thinker.prompt_token_ids,
            thinker_output_token_ids=[thinker.output_token_ids[0]],
            thinker_prefill_embed=thinker.embed_table[: prefill_len + 1],
            thinker_prefill_hidden=thinker.hidden_table[: prefill_len + 1],
            tts_bos_embed_thinker=thinker.tts_bos_embed,
            tts_eos_embed_thinker=thinker.tts_eos_embed,
            tts_pad_embed_thinker=thinker.tts_pad_embed,
            speaker=self.cfg.speaker,
        )

        # Push every remaining decode-token embedding into the trailing
        # queue at once. The thinker's embed_table has at most P + N - 1
        # rows (the most-recently-sampled token's embedding isn't
        # computed until the next decode forward), so the last sampled
        # token's embedding is intentionally absent -- a negligible loss
        # of one trailing slot
        remaining_start = prefill_len + 1
        # vLLM-Omni's async Talker handoff is one token behind the
        # Thinker stream: the final Thinker token is never used as a
        # Talker text-conditioning slot. Single-stage Thinker outputs are
        # cumulative and may already contain that final row, so cap by
        # len(output_token_ids) - 1 to preserve upstream behavior.
        available_embeds = min(
            embed_rows - prefill_len,
            max(len(thinker.output_token_ids) - 1, 0),
        )
        remaining_end = prefill_len + available_embeds
        if remaining_end > remaining_start:
            remaining_embeds = thinker.embed_table[remaining_start:remaining_end]
            self.talker_runner.append_thinker_decode_token(remaining_embeds)

        # No more thinker tokens are coming: the runner will fall back
        # to tts_eos / tts_pad once the trailing queue is exhausted.
        self.talker_runner.mark_thinker_finished()

        codec_frames: List[torch.Tensor] = []
        t_start = time.time()
        while True:
            frame = self.talker_runner.step()
            if frame is None:
                if not self.talker_runner.is_done():
                    # The runner reports None only when the trailing
                    # queue is empty AND the thinker session isn't
                    # finished -- but we just finished it. Bail rather
                    # than spin.
                    raise RuntimeError(
                        "Talker step() returned None but is_done()=False; "
                        "trailing queue underrun?"
                    )
                break
            codec_frames.append(frame)

        logger.info(
            "Talker finished: %d codec frames in %.2fs",
            len(codec_frames), time.time() - t_start,
        )
        return codec_frames

    async def _run_code2wav_to_completion(
        self,
        codec_frames: List[torch.Tensor],
        request_id: str,
    ) -> Tuple[np.ndarray, int]:
        """Vocode the entire codec sequence in a single Code2Wav request.

        Because we submit one prompt covering all frames there are no
        chunk boundaries, so no left-context handling is needed and we
        simply concatenate every audio chunk the engine yields.
        """
        if not codec_frames:
            return np.empty(0, dtype=np.float32), int(self.cfg.audio_sample_rate)

        chunk_codes = torch.stack(codec_frames, dim=0)
        prompt = _build_code2wav_prompt(chunk_codes)
        sp = SamplingParams(
            max_tokens=self.cfg.sampling.code2wav_max_tokens,
            temperature=self.cfg.sampling.code2wav_temperature,
            top_p=self.cfg.sampling.code2wav_top_p,
            repetition_penalty=self.cfg.sampling.code2wav_repetition_penalty,
            seed=self.cfg.seed,
            detokenize=True,
        )

        consumed = 0
        sample_rate = int(self.cfg.audio_sample_rate)
        chunks: List[np.ndarray] = []
        t_submit = time.time()
        async for omni_out in self.code2wav_engine.generate(
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

        if not chunks:
            audio = np.empty(0, dtype=np.float32)
        else:
            audio = np.concatenate(chunks).astype(np.float32, copy=False)
        logger.info(
            "Code2Wav finished: %d codec frames -> %d samples (%.2fs total)",
            len(codec_frames), audio.size, time.time() - t_submit,
        )
        return audio, sample_rate

    def _write_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """Slice ``audio`` into fixed-size frames and push to the output buffer.

        The trailing partial frame is zero-padded so consumers always
        see fixed-size frames.
        """
        if int(sample_rate) != int(self.cfg.audio_sample_rate):
            # Re-publish the rate so the adapter knows what to play back.
            self.audio_meta_buffer.send(int(sample_rate), timeout_s=0.05)

        if audio.size == 0:
            return

        frame_samples = int(self.cfg.audio_frame_samples)
        offset = 0
        while audio.size - offset >= frame_samples:
            self.audio_output_buffer.write(
                audio[offset:offset + frame_samples].astype(np.float32, copy=False),
            )
            offset += frame_samples

        rem = audio.size - offset
        if rem > 0:
            tail = np.zeros(frame_samples, dtype=np.float32)
            tail[:rem] = audio[offset:]
            self.audio_output_buffer.write(tail)

    def _handle_prompt(self, user_text: str) -> None:
        """Run thinker -> talker -> code2wav -> output, one stage at a time."""
        request_id = str(uuid.uuid4())
        t0 = time.time()
        try:
            thinker = self._async_runner.run(
                self._run_thinker_to_completion(user_text, f"{request_id}-thinker"),
            )
            codec_frames = self._run_talker_to_completion(thinker)
            audio, sr = self._async_runner.run(
                self._run_code2wav_to_completion(codec_frames, f"{request_id}-c2w"),
            )
            self._write_audio(audio, sr)
            logger.info(
                "Total response time: %.2fs (sequential thinker->talker->c2w)",
                time.time() - t0,
            )
        except Exception:
            logger.exception("Qwen3-Omni inference failed")

    # ------------------------------------------------------------------
    # control buffer + main loop
    # ------------------------------------------------------------------

    def is_start(self) -> bool:
        return int(self.ctrl_buffer.recv()) == self._CMD_START

    def is_terminate(self) -> bool:
        return int(self.ctrl_buffer.recv()) == self._CMD_TERMINATE

    def is_reset(self) -> bool:
        return int(self.ctrl_buffer.recv()) == self._CMD_RESET

    def loop(self) -> None:
        try:
            while True:
                if self.is_terminate():
                    self.terminate()
                    break

                if self.is_start() and not self.session_started:
                    self.start()
                elif self.is_reset() and self.session_started:
                    self.reset()

                if not self.session_started:
                    time.sleep(0.005)
                    continue

                user_text = self._read_pending_text()
                if user_text is not None and user_text.strip():
                    logger.info("Qwen3-Omni handling prompt: %r", user_text[:80])
                    self._handle_prompt(user_text)
                else:
                    time.sleep(0.001)
        except KeyboardInterrupt:
            logger.info("Qwen3-Omni worker interrupted, terminating...")
            self.terminate()
