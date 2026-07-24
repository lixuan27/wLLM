"""Streaming Qwen3-Omni worker.

Same adapter contract as the reference (creates the ctrl / text_input /
audio_output / audio_meta shm buffers under the config's names; responds
to start/terminate/reset; emits 960-sample float32 frames). The schedule
is parameterized by StreamingParams so one worker backs several variants
(each a distinct config + launch + benchmark, never sharing a measurement):

  * stream_c2w           : reference Thinker->Talker (full), then stream
                           Talker->Code2Wav (growing-prefix vocoding,
                           audio emitted incrementally).
  * stream_thinker_talker: prime the Talker after the Thinker's first
                           token and push decode embeds as they stream
                           (overlaps Thinker decode with Talker), full
                           Code2Wav at the end.
  * stream_full          : both.

Concurrency model (deadlock-free): ONE persistent main-thread asyncio
loop drives BOTH AsyncOmni engines (thinker, code2wav) for warmup AND
every prompt -> each engine's janus queues bind to a single loop. The
synchronous Talker step() is offloaded to a dedicated single-thread
executor via run_in_executor and awaited, so the thinker async generator
(streaming tokens into the Talker trailing queue) and the code2wav
vocode coroutines interleave on the same loop. The Talker runner is
lock-safe for concurrent append (no lock) + step (locked).

Correctness: the Talker is unchanged and deterministic, so the emitted
CODEC FRAMES are bit-identical to the reference. The audio differs from
the reference's single full-sequence decode only by the vocoder's
non-causal chunk-boundary effect (the vocoder is globally non-causal),
bounded by growing-prefix vocoding.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import List, Optional

import numpy as np
import torch

from wllm.apps.qwen3_omni.adapter import TEXT_FRAME_BYTES, decode_text_frame
from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.logger import init_logger
from wllm.serving.utils.rand import set_global_seed

from wllm.apps.qwen3_omni.backend.cuda import blocks
from wllm.apps.qwen3_omni.backend.cuda.streaming.config import StreamingConfig, StreamingParams

logger = init_logger(__name__)


class _AudioWriter:
    """Frame a stream of samples into fixed 960-sample frames; pad only the
    final partial frame (matches the reference _write_audio framing)."""

    def __init__(self, out_buf, frame_samples):
        self.out = out_buf
        self.fs = int(frame_samples)
        self.rem = np.empty(0, dtype=np.float32)

    def push(self, samples: np.ndarray):
        if samples.size == 0:
            return
        buf = np.concatenate([self.rem, samples.astype(np.float32, copy=False)])
        n_full = buf.shape[0] // self.fs
        if n_full > 0:
            self.out.write(buf[: n_full * self.fs].reshape(n_full, self.fs))
        self.rem = buf[n_full * self.fs:]

    def flush_pad(self):
        if self.rem.size > 0:
            tail = np.zeros(self.fs, dtype=np.float32)
            tail[: self.rem.size] = self.rem
            self.out.write(tail)
            self.rem = np.empty(0, dtype=np.float32)


class StreamingWorker:
    _CMD_START = 1
    _CMD_TERMINATE = 2
    _CMD_RESET = 3

    def __init__(self, cfg_path: str):
        self.scfg = StreamingConfig.from_yaml(cfg_path)
        self.cfg = self.scfg.base
        self.sp: StreamingParams = self.scfg.streaming
        set_global_seed(self.cfg.seed)
        self._init_buffers()
        self._runner = asyncio.Runner()           # the single persistent loop
        self._init_engines()
        self.session_started = False
        self._next_text_index = 0
        self.audio_meta_buffer.send(int(self.cfg.audio_sample_rate), timeout_s=10.0)
        logger.info("serving: streaming worker sp=%s", self.sp)
        logger.info("Qwen3-Omni backend READY")

    # ------------------------------------------------------------------
    def _init_buffers(self):
        self.ctrl_buffer = SharedControlBuffer(self.cfg.ctrl_buffer_name, create=True)
        self.text_input_buffer = SharedTensorBuffer(
            name=self.cfg.text_input_buffer_name, frame_shape=(TEXT_FRAME_BYTES,),
            max_len=int(self.cfg.text_max_pending), dtype=np.uint8, create=True)
        self.audio_output_buffer = SharedTensorBuffer(
            self.cfg.audio_output_buffer_name,
            frame_shape=(int(self.cfg.audio_frame_samples),), dtype=np.float32,
            max_len=int(self.cfg.audio_max_chunks), create=True)
        self.audio_meta_buffer = SharedControlBuffer(self.cfg.audio_meta_buffer_name, create=True)

    def _init_engines(self):
        sp = self.sp
        # NOTE on device placement with a TP thinker: a TP AsyncOmni
        # thinker's mp executor initializes CUDA in the PARENT process while
        # CUDA_VISIBLE_DEVICES is pinned to the thinker's GPUs, which
        # PERMANENTLY locks the parent context to exactly those GPUs (a CUDA
        # context cannot be widened after init). So the thinker_tp* variants
        # launch the worker with CVD == the thinker's GPU set and place the
        # in-process Talker + Code2Wav on ONE of those GPUs (each 143 GB
        # H200 has room for a thinker shard + the small talker/c2w). The
        # lock is then consistent and the Talker's cuda:N is valid. The
        # Talker also runs eager (talker_enforce_eager) so torch.compile
        # device-id cache entries don't collide with the TP thinker's.
        self.thinker_engine = blocks.make_thinker_engine(
            self.cfg, visible_devices=sp.thinker_visible_devices,
            stage_configs_path=sp.thinker_stage_configs_path)
        if sp.talker_enforce_eager and sp.talker_tp_size <= 1:
            # eager talker (no torch.compile) to coexist with a TP thinker
            from wllm.apps.qwen3_omni.backend.cuda.talker_tp.runner_tp import Qwen3OmniTalkerRunner as _EagerRunner
            cfg = self.cfg
            self.talker_runner = _EagerRunner(
                cfg.talker.model_path,
                gpu_index=int(cfg.talker.gpu_index) if cfg.talker.gpu_index is not None else 0,
                temperature=cfg.sampling.talker_temperature, top_k=cfg.sampling.talker_top_k,
                top_p=cfg.sampling.talker_top_p, repetition_penalty=cfg.sampling.talker_repetition_penalty,
                seed=cfg.seed, max_tokens=cfg.sampling.talker_max_tokens,
                max_seq_len=cfg.sampling.talker_max_seq_len, enforce_eager=True)
        else:
            self.talker_runner = blocks.make_talker_runner(self.cfg)
        self.c2w_engine = blocks.make_c2w_engine(
            self.cfg, visible_devices=sp.c2w_visible_devices,
            stage_configs_path=sp.c2w_stage_configs_path)
        try:
            self._runner.run(self._warmup_c2w())
        except Exception:
            logger.exception("c2w warmup failed")

    async def _warmup_c2w(self):
        dummy = [torch.zeros(16, dtype=torch.long) for _ in range(8)]
        await blocks._c2w_collect(self.c2w_engine, self.cfg, dummy, "warmup-c2w-0")

    # ------------------------------------------------------------------
    def start(self):
        self.session_started = True
        self.audio_output_buffer.clear()
        self._next_text_index = self.text_input_buffer.num
        self.ctrl_buffer.commit()
        logger.info("session started")

    def reset(self):
        self.session_started = False
        self.audio_output_buffer.clear()
        self._next_text_index = self.text_input_buffer.num
        self.ctrl_buffer.commit()

    def terminate(self):
        self.session_started = False
        for name, eng in (("thinker", getattr(self, "thinker_engine", None)),
                          ("code2wav", getattr(self, "c2w_engine", None))):
            try:
                if eng is not None:
                    eng.shutdown()
            except Exception:
                logger.exception("error shutting down %s", name)
        try:
            if getattr(self, "talker_runner", None) is not None:
                self.talker_runner.shutdown()
        except Exception:
            pass
        try:
            self._runner.close()
        except Exception:
            pass
        self.ctrl_buffer.commit()
        for buf in (self.ctrl_buffer, self.text_input_buffer,
                    self.audio_output_buffer, self.audio_meta_buffer):
            try:
                buf.unlink()
            except Exception:
                pass
        logger.info("worker terminated")

    # ------------------------------------------------------------------
    def _read_pending_text(self) -> Optional[str]:
        next_idx, frame = self.text_input_buffer.read(self._next_text_index, 1)
        self._next_text_index = next_idx
        if frame is None:
            return None
        return decode_text_frame(frame[0])

    # ------------------------------------------------------------------
    # Thinker streaming feeder (runs as a task on the main loop).
    async def _thinker_feed(self, user_text: str, req_id: str, primed_evt: asyncio.Event):
        from wllm.apps.qwen3_omni.reference.worker import (
            _format_chat_prompt, _coalesce, _take_first_marker,
            _iter_completion_outputs, ThinkerOutput,
        )
        cfg, runner = self.cfg, self.talker_runner
        prompt = {"prompt": _format_chat_prompt(cfg.system_prompt, user_text)}
        sp = blocks.thinker_sampling_params(cfg)
        primed = False
        P = 0
        pushed = 0  # decode-embed rows beyond (P+1) pushed so far
        async for out in self.thinker_engine.generate(
                prompt=prompt, request_id=req_id,
                sampling_params_list=[sp], output_modalities=["text"]):
            completion = next(iter(_iter_completion_outputs(out)), None)
            if completion is None:
                continue
            mm = getattr(completion, "multimodal_output", None) or {}
            embed = _coalesce(mm.get("0"))
            hidden = _coalesce(mm.get("24"))
            req = getattr(out, "request_output", None)
            ptoks = list(getattr(req, "prompt_token_ids", []) or [])
            otoks = list(getattr(completion, "token_ids", []) or [])
            tts_bos = _take_first_marker(mm.get("tts_bos_embed"))
            tts_eos = _take_first_marker(mm.get("tts_eos_embed"))
            tts_pad = _take_first_marker(mm.get("tts_pad_embed"))
            if (embed is None or hidden is None or not ptoks or not otoks
                    or tts_bos is None or tts_eos is None or tts_pad is None):
                continue
            if not primed:
                P = len(ptoks)
                if embed.shape[0] < P + 1:
                    continue
                tout = ThinkerOutput(
                    prompt_token_ids=ptoks, output_token_ids=[otoks[0]],
                    embed_table=embed, hidden_table=hidden,
                    tts_bos_embed=tts_bos, tts_eos_embed=tts_eos, tts_pad_embed=tts_pad,
                    text="")
                blocks.prime_talker(runner, tout, cfg, push_all=False)
                primed = True
                primed_evt.set()
            # rows [P+1 .. P+avail) are safe; cap so the final handoff-excluded
            # token is never pushed (matches blocks.prime_talker push_all logic).
            avail = min(embed.shape[0] - (P + 1), max(len(otoks) - 1 - 1, 0))
            if avail > pushed:
                new_rows = embed[P + 1 + pushed: P + 1 + avail]
                if new_rows.shape[0] > 0:
                    runner.append_thinker_decode_token(new_rows)
                    pushed = avail
        if not primed:
            primed_evt.set()
            raise RuntimeError("thinker produced no usable yield to prime talker")
        runner.mark_thinker_finished()

    # ------------------------------------------------------------------
    async def _vocode_prefix(self, acc: List[torch.Tensor], start: int, end: int, req_id: str):
        decode_frames = acc[start:end]
        if not decode_frames:
            return np.empty(0, dtype=np.float32)
        audio, _ = await blocks._c2w_collect(self.c2w_engine, self.cfg, decode_frames,
                                             f"{req_id}-c2w-{end}")
        return audio

    async def _run_prompt(self, user_text: str, req_id: str):
        sp = self.sp
        runner = self.talker_runner
        writer = _AudioWriter(self.audio_output_buffer, self.cfg.audio_frame_samples)

        thinker_task = None
        if sp.stream_thinker_talker:
            runner._reset_session_state()
            primed_evt = asyncio.Event()
            thinker_task = asyncio.create_task(
                self._thinker_feed(user_text, f"{req_id}-thinker", primed_evt))
            await primed_evt.wait()
            if thinker_task.done():  # raised before priming
                await thinker_task
        else:
            thinker = await blocks._thinker_collect(
                self.thinker_engine, self.cfg, user_text, f"{req_id}-thinker")
            blocks.prime_talker(runner, thinker, self.cfg, push_all=True)

        acc: List[torch.Tensor] = []
        emitted_samples = 0
        last_emit_end = 0
        all_frames: List[torch.Tensor] = []

        spf = int(self.cfg.audio_samples_per_codec_frame)

        async def emit(final: bool):
            # Unified growing/windowed emission. The vocoder emits
            # n_frames*spf - TRIM samples with the fixed trim at the TAIL
            # (probe_c2w2.py: full[f*spf:(f+1)*spf] aligns to the window
            # decode's local frame f), so a window over global frames
            # [start:end] has its local sample i == global sample start*spf + i.
            # We track emitted_samples in GLOBAL sample coordinates and emit
            # only the not-yet-emitted tail.
            nonlocal emitted_samples, last_emit_end
            n = len(acc)
            end = n if final else max(0, n - sp.c2w_lookahead_frames)
            if end <= last_emit_end and not final:
                return
            if sp.c2w_context_frames is not None and sp.c2w_context_frames >= 0 and not final:
                start = max(0, end - sp.c2w_context_frames)  # bounded left context
            else:
                start = 0  # growing prefix (default; best fidelity)
            audio = await self._vocode_prefix(acc, start, end, req_id)
            local_emitted = max(0, emitted_samples - start * spf)
            new = audio[local_emitted:]
            emitted_samples = max(emitted_samples, start * spf + audio.shape[0])
            writer.push(new)
            last_emit_end = end

        # Step talker on THIS (main) thread -- the thread it was warmed up
        # on, so its CUDA graphs replay correctly. Yield to the loop between
        # steps so the background thinker-feed task and c2w vocode coroutines
        # interleave (thinker subprocess + talker GPU + c2w engine overlap).
        if sp.stream_c2w and sp.c2w_threaded:
            # Decoupled c2w: a concurrent pump coroutine vocodes/emits while
            # the talker keeps stepping, so the talker never blocks on a
            # (growing) c2w vocode. Same single event loop -> no cross-loop
            # deadlock; emit state is owned solely by the pump.
            talker_done = asyncio.Event()
            new_frame = asyncio.Event()

            async def _pump():
                while True:
                    await new_frame.wait()
                    new_frame.clear()
                    while (len(acc) >= sp.c2w_first_emit_frames
                           and (len(acc) - last_emit_end) >= sp.c2w_emit_interval_frames):
                        await emit(final=False)
                    if talker_done.is_set():
                        if len(acc) > last_emit_end:
                            await emit(final=True)
                        break

            pump_task = asyncio.create_task(_pump())
            while True:
                frame = runner.step()
                if frame is None:
                    if runner.is_done():
                        break
                    await asyncio.sleep(0.001)
                    continue
                acc.append(frame)
                all_frames.append(frame)
                new_frame.set()
                await asyncio.sleep(0)
            talker_done.set()
            new_frame.set()
            await pump_task
            writer.flush_pad()
        elif sp.stream_c2w:
            while True:
                frame = runner.step()
                if frame is None:
                    if runner.is_done():
                        break
                    await asyncio.sleep(0.001)  # waiting on thinker tokens
                    continue
                acc.append(frame)
                all_frames.append(frame)
                n = len(acc)
                if n >= sp.c2w_first_emit_frames and (n - last_emit_end) >= sp.c2w_emit_interval_frames:
                    await emit(final=False)
                await asyncio.sleep(0)
            await emit(final=True)
            writer.flush_pad()
        else:
            while True:
                frame = runner.step()
                if frame is None:
                    if runner.is_done():
                        break
                    await asyncio.sleep(0.001)
                    continue
                all_frames.append(frame)
                await asyncio.sleep(0)
            audio, _ = await blocks._c2w_collect(self.c2w_engine, self.cfg, all_frames,
                                                 f"{req_id}-c2w")
            writer.push(np.asarray(audio, dtype=np.float32).reshape(-1))
            writer.flush_pad()

        if thinker_task is not None:
            await thinker_task
        return len(all_frames)

    def _handle_prompt(self, user_text: str):
        req_id = str(uuid.uuid4())
        t0 = time.time()
        try:
            n = self._runner.run(self._run_prompt(user_text, req_id))
            logger.info("prompt done in %.2fs (%d frames)", time.time() - t0, n)
        except Exception:
            logger.exception("streaming inference failed")

    # ------------------------------------------------------------------
    def is_cmd(self, c):
        return int(self.ctrl_buffer.recv()) == c

    def loop(self):
        try:
            while True:
                if self.is_cmd(self._CMD_TERMINATE):
                    self.terminate()
                    break
                if self.is_cmd(self._CMD_START) and not self.session_started:
                    self.start()
                elif self.is_cmd(self._CMD_RESET) and self.session_started:
                    self.reset()
                if not self.session_started:
                    time.sleep(0.005)
                    continue
                txt = self._read_pending_text()
                if txt is not None and txt.strip():
                    logger.info("handling prompt: %r", txt[:80])
                    self._handle_prompt(txt)
                else:
                    time.sleep(0.001)
        except KeyboardInterrupt:
            self.terminate()
