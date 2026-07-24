"""Streaming / pipelined Qwen3-Omni worker.

Honors the shared adapter contract (same shm buffers, same control opcodes)
as the reference backend, but replaces the reference's strictly-sequential
`_handle_prompt` with a scheduling mode chosen by config:

  * sequential           — reference-faithful (thinker -> talker -> c2w whole).
  * stream_talker_c2w    — thinker whole; talker->c2w chunked & overlapped.
  * stream_thinker_talker— thinker streams into talker (overlapped); c2w whole.
  * full_stream          — all three stages overlapped; c2w chunked.

The streaming transformations are the ones the IR analysis surfaced:
- thinker->talker variable-rate streaming overlap (talker consumes decode
  embeds incrementally; bit-identical codec frames — each step needs only
  embed[gen_step]).
- talker->code2wav fixed-rate streaming overlap (vocode fixed-size codec
  chunks with left context as they arrive; near-identical audio vs whole
  sequence, and avoids per-shape MIOpen re-search).

Concurrency: the thinker and code2wav are AsyncOmni engines living in their
own subprocesses (the worker just does I/O against them, in an asyncio
loop). The talker forward is the only heavy in-process compute, so it runs
in its own thread and communicates via thread-safe queues — exactly the
threading model the vendored runner was designed for (append from one
thread while step runs in another).
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
import uuid
from typing import List, Optional

import numpy as np
import torch

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.logger import init_logger
from wllm.apps.qwen3_omni.adapter import TEXT_FRAME_BYTES, decode_text_frame
from wllm.serving.utils.rand import set_global_seed
from wllm.serving.utils.torch_utils import set_torch_options

from wllm.apps.qwen3_omni.backend.rocm.streaming.config import StreamingConfig
from wllm.apps.qwen3_omni.backend.rocm.pipeline.components import (
    ThinkerComponent, TalkerComponent, Code2WavComponent,
    ThinkerCfg, TalkerCfg, Code2WavCfg)

logger = init_logger(__name__)
set_torch_options()


class _AudioWriter:
    """Accumulates streamed audio and writes fixed-size 960-sample frames to
    the output buffer with NO inter-chunk gaps (carries a residual across
    chunks; zero-pads only the final flush)."""

    def __init__(self, out_buffer: SharedTensorBuffer, frame_samples: int):
        self.buf = out_buffer
        self.frame_samples = int(frame_samples)
        self._residual = np.empty(0, dtype=np.float32)
        self.frames_written = 0

    def push(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            return
        data = np.concatenate([self._residual, audio.astype(np.float32, copy=False)])
        n_full = data.size // self.frame_samples
        if n_full > 0:
            block = data[: n_full * self.frame_samples].reshape(n_full, self.frame_samples)
            self.buf.write(block)
            self.frames_written += n_full
        self._residual = data[n_full * self.frame_samples:]

    def flush(self) -> None:
        if self._residual.size > 0:
            tail = np.zeros(self.frame_samples, dtype=np.float32)
            tail[: self._residual.size] = self._residual
            self.buf.write(tail)
            self.frames_written += 1
            self._residual = np.empty(0, dtype=np.float32)


class StreamingWorker:
    _CMD_START = 1
    _CMD_TERMINATE = 2
    _CMD_RESET = 3

    def __init__(self, cfg_path: str):
        self.cfg = StreamingConfig.from_yaml(cfg_path)
        self._init_worker()
        logger.info("serving: streaming worker mode=%s", self.cfg.pipeline_mode)
        logger.info("Qwen3-Omni backend READY")

    # ------------------------------------------------------------------
    def _init_worker(self):
        c = self.cfg
        set_global_seed(c.seed)
        self.ctrl_buffer = SharedControlBuffer(c.ctrl_buffer_name, create=True)
        self.text_input_buffer = SharedTensorBuffer(
            name=c.text_input_buffer_name, frame_shape=(TEXT_FRAME_BYTES,),
            max_len=int(c.text_max_pending), dtype=np.uint8, create=True)
        self.audio_output_buffer = SharedTensorBuffer(
            c.audio_output_buffer_name, frame_shape=(int(c.audio_frame_samples),),
            dtype=np.float32, max_len=int(c.audio_max_chunks), create=True)
        self.audio_meta_buffer = SharedControlBuffer(c.audio_meta_buffer_name, create=True)

        self._load_components()

        self._async_runner = asyncio.Runner()
        self.session_started = False
        self._next_text_index = 0
        self.audio_meta_buffer.send(int(c.audio_sample_rate), timeout_s=10.0)

        # Warm up code2wav lazy init AND the MIOpen conv kernels for EVERY
        # chunk shape the streaming schedule can emit, so measured requests
        # never pay a per-shape kernel search (a slow chunk causes a real-time
        # underrun — observed on short responses whose variable FINAL chunk hit
        # an un-warmed shape). Shapes:
        #   - chunk 0 (no left context): 1..first_chunk_frames
        #   - steady/final chunk (+ left context): (1+lc)..(chunk_frames+lc)
        if c.pipeline_mode != "sequential":
            try:
                lc = int(c.codec_left_context_frames)
                shapes = set()
                shapes.update(range(1, int(c.first_chunk_frames) + 1))
                shapes.update(range(1 + lc, int(c.codec_chunk_frames) + lc + 1))
                shapes.discard(0)
                for i, nf in enumerate(sorted(shapes)):
                    self._async_runner.run(self.code2wav.warmup(
                        n_frames=nf, request_id=f"warmup-c2w-{i}"))
                logger.info("code2wav warmup: %d chunk shapes", len(shapes))
            except Exception:
                logger.exception("code2wav warmup failed")
        else:
            try:
                self._async_runner.run(self.code2wav.warmup(n_frames=8))
            except Exception:
                logger.exception("code2wav warmup failed")

    def _load_components(self):
        c = self.cfg
        self.thinker = ThinkerComponent(
            ThinkerCfg(c.model_path, c.thinker_stage_configs_path,
                       gpu_index=c.thinker_gpu, tensor_parallel_size=c.thinker_tp,
                       max_tokens=c.thinker_max_tokens, temperature=c.thinker_temperature,
                       top_p=c.thinker_top_p, top_k=c.thinker_top_k,
                       repetition_penalty=c.thinker_repetition_penalty, seed=c.seed),
            c.system_prompt)
        self.talker = TalkerComponent(
            TalkerCfg(c.model_path, gpu_index=c.talker_gpu, tensor_parallel_size=c.talker_tp,
                      temperature=c.talker_temperature, top_k=c.talker_top_k,
                      top_p=c.talker_top_p, repetition_penalty=c.talker_repetition_penalty,
                      max_tokens=c.talker_max_tokens, max_seq_len=c.talker_max_seq_len,
                      seed=c.seed), c.speaker)
        self.code2wav = Code2WavComponent(
            Code2WavCfg(c.model_path, c.code2wav_stage_configs_path,
                        gpu_index=c.c2w_gpu, tensor_parallel_size=c.c2w_tp,
                        max_tokens=c.code2wav_max_tokens, temperature=c.code2wav_temperature,
                        top_p=c.code2wav_top_p, repetition_penalty=c.code2wav_repetition_penalty,
                        seed=c.seed))

    # ------------------------------------------------------------------
    # session lifecycle (matches reference)
    # ------------------------------------------------------------------
    def start(self):
        self.session_started = True
        self.audio_output_buffer.clear()
        self._next_text_index = self.text_input_buffer.num
        self.ctrl_buffer.commit()
        logger.info("Qwen3-Omni streaming session started")

    def reset(self):
        self.session_started = False
        self.audio_output_buffer.clear()
        self._next_text_index = self.text_input_buffer.num
        self.ctrl_buffer.commit()

    def terminate(self):
        self.session_started = False
        for comp in (getattr(self, "thinker", None), getattr(self, "code2wav", None)):
            if comp is not None:
                comp.shutdown()
        if getattr(self, "talker", None) is not None:
            self.talker.shutdown()
            self.talker = None
        if getattr(self, "_async_runner", None) is not None:
            try:
                self._async_runner.close()
            except Exception:
                pass
            self._async_runner = None
        self.ctrl_buffer.commit()
        for buf, name in ((self.ctrl_buffer, "ctrl"), (self.text_input_buffer, "text"),
                          (self.audio_output_buffer, "audio"), (self.audio_meta_buffer, "meta")):
            try:
                buf.unlink()
            except Exception:
                logger.exception("error unlinking %s", name)
        logger.info("Qwen3-Omni streaming worker terminated")

    # ------------------------------------------------------------------
    # prompt handling — dispatch on mode
    # ------------------------------------------------------------------
    def _handle_prompt(self, user_text: str) -> None:
        rid = str(uuid.uuid4())
        t0 = time.time()
        try:
            mode = self.cfg.pipeline_mode
            writer = _AudioWriter(self.audio_output_buffer, self.cfg.audio_frame_samples)
            if mode == "sequential":
                self._async_runner.run(self._run_sequential(user_text, rid, writer))
            elif mode == "stream_talker_c2w":
                self._async_runner.run(self._run_stream_talker_c2w(user_text, rid, writer))
            elif mode == "stream_thinker_talker":
                self._async_runner.run(self._run_stream_thinker_talker(user_text, rid, writer))
            elif mode == "full_stream":
                self._async_runner.run(self._run_full_stream(user_text, rid, writer))
            else:
                raise ValueError(f"unknown pipeline_mode {mode!r}")
            writer.flush()
            logger.info("Total response time: %.2fs (mode=%s, %d audio frames)",
                        time.time() - t0, mode, writer.frames_written)
        except Exception:
            logger.exception("streaming inference failed")

    # -------- mode: sequential (reference-faithful) --------
    async def _run_sequential(self, user_text, rid, writer):
        to = await self.thinker.run_to_completion(user_text, f"{rid}-thinker")
        frames = self.talker.run_to_completion(to)
        audio, sr = await self.code2wav.vocode(frames, f"{rid}-c2w", self.cfg.audio_sample_rate)
        self._publish_sr(sr)
        writer.push(audio)

    # -------- mode: stream_thinker_talker (c2w whole) --------
    async def _run_stream_thinker_talker(self, user_text, rid, writer):
        frames = await self._thinker_talker_stream(user_text, rid, collect=True)
        audio, sr = await self.code2wav.vocode(frames, f"{rid}-c2w", self.cfg.audio_sample_rate)
        self._publish_sr(sr)
        writer.push(audio)

    # -------- mode: stream_talker_c2w (thinker whole) --------
    async def _run_stream_talker_c2w(self, user_text, rid, writer):
        to = await self.thinker.run_to_completion(user_text, f"{rid}-thinker")
        self.talker.prime_whole(to)
        await self._talker_c2w_stream(rid, writer, thinker_done_event=None)

    # -------- mode: full_stream (all overlapped) --------
    async def _run_full_stream(self, user_text, rid, writer):
        # thinker streams into the talker (priming + incremental appends) while
        # the talker thread steps and the c2w consumes chunks.
        primed = threading.Event()
        thinker_done = threading.Event()
        thinker_task = asyncio.create_task(
            self._thinker_feed(user_text, rid, primed, thinker_done))
        # wait until the talker is primed before starting the talker thread
        while not primed.is_set() and not thinker_task.done():
            await asyncio.sleep(0.002)
        if thinker_task.done():
            thinker_task.result()  # surface any exception
        await self._talker_c2w_stream(rid, writer, thinker_done_event=thinker_done)
        await thinker_task

    # ------------------------------------------------------------------
    # streaming building blocks
    # ------------------------------------------------------------------
    async def _thinker_feed(self, user_text, rid, primed: threading.Event,
                            thinker_done: threading.Event):
        """Stream thinker embeds into the talker (prime once, then append)."""
        r = self.talker.runner
        appended_upto = 0
        P = None
        try:
            async for (embed, hidden, ptids, otids, markers, is_final, to) in \
                    self.thinker.stream(user_text, f"{rid}-thinker"):
                if P is None and ptids:
                    P = len(ptids)
                if not primed.is_set():
                    if P is not None and markers is not None and embed.shape[0] >= P + 1 and otids:
                        bos, eos, pad = markers
                        r._reset_session_state()
                        r.start_session(
                            thinker_prompt_token_ids=ptids,
                            thinker_output_token_ids=[otids[0]],
                            thinker_prefill_embed=embed[: P + 1],
                            thinker_prefill_hidden=hidden[: P + 1],
                            tts_bos_embed_thinker=bos, tts_eos_embed_thinker=eos,
                            tts_pad_embed_thinker=pad, speaker=self.cfg.speaker)
                        appended_upto = P + 1
                        primed.set()
                    else:
                        continue
                # append any new trailing embeds up to the reference cap
                target = min(embed.shape[0], P + max(len(otids) - 1, 0))
                if target > appended_upto:
                    r.append_thinker_decode_token(embed[appended_upto:target])
                    appended_upto = target
        finally:
            # always mark finished so the talker thread can drain (tts_eos/pad)
            # instead of stalling forever waiting for more embeds.
            if primed.is_set():
                r.mark_thinker_finished()
            thinker_done.set()

    def _thinker_talker_stream(self, user_text, rid, collect=False):
        """Return a coroutine that streams thinker->talker and collects all
        codec frames (used by stream_thinker_talker mode)."""
        async def _run():
            primed = threading.Event()
            thinker_done = threading.Event()
            frames: List[torch.Tensor] = []
            thinker_task = asyncio.create_task(
                self._thinker_feed(user_text, rid, primed, thinker_done))
            while not primed.is_set() and not thinker_task.done():
                await asyncio.sleep(0.002)
            if thinker_task.done():
                thinker_task.result()

            def talker_loop():
                r = self.talker.runner
                torch.cuda.set_device(r.device)  # current device is thread-local
                while True:
                    frame = r.step()
                    if frame is None:
                        if r.is_done():
                            break
                        time.sleep(0.001)
                        continue
                    frames.append(frame)

            t = threading.Thread(target=talker_loop, daemon=True)
            t.start()
            await thinker_task
            await asyncio.get_event_loop().run_in_executor(None, t.join)
            return frames
        return _run()

    async def _talker_c2w_stream(self, rid, writer, thinker_done_event):
        """Run the talker step loop in a thread; consume codec frames in fixed
        chunks (with left context) and vocode+write each chunk as it fills."""
        codec_q: "queue.Queue" = queue.Queue()

        def talker_loop():
            r = self.talker.runner
            torch.cuda.set_device(r.device)  # current device is thread-local
            while True:
                frame = r.step()
                if frame is None:
                    if r.is_done():
                        break
                    time.sleep(0.001)   # stall: waiting for thinker embeds
                    continue
                codec_q.put(frame)
            codec_q.put(None)  # sentinel

        t = threading.Thread(target=talker_loop, daemon=True)
        t.start()

        c = self.cfg
        spf = c.audio_samples_per_codec_frame
        left_ctx = c.codec_left_context_frames
        chunk_idx = 0
        pending: List[torch.Tensor] = []       # new frames not yet vocoded
        context: List[torch.Tensor] = []        # left-context frames
        loop = asyncio.get_event_loop()
        sr_published = [False]

        async def emit(take: List[torch.Tensor]):
            nonlocal context, chunk_idx
            window = context + take
            audio, sr = await self.code2wav.vocode_chunk(
                window, len(take), spf, f"{rid}-c2w-{chunk_idx}", c.audio_sample_rate)
            if not sr_published[0]:
                self._publish_sr(sr); sr_published[0] = True
            writer.push(audio)
            context = (context + take)[-left_ctx:] if left_ctx > 0 else []
            chunk_idx += 1

        async def maybe_flush(is_final: bool):
            nonlocal pending
            while pending:
                target = c.first_chunk_frames if chunk_idx == 0 else c.codec_chunk_frames
                if is_final:
                    take, pending = pending, []
                elif len(pending) >= target:
                    take, pending = pending[:target], pending[target:]
                else:
                    return
                await emit(take)

        while True:
            frame = await loop.run_in_executor(None, codec_q.get)
            if frame is None:
                break
            pending.append(frame)
            await maybe_flush(is_final=False)
        await maybe_flush(is_final=True)
        await loop.run_in_executor(None, t.join)

    def _publish_sr(self, sr: int):
        if int(sr) != int(self.cfg.audio_sample_rate):
            self.audio_meta_buffer.send(int(sr), timeout_s=0.05)

    # ------------------------------------------------------------------
    # control loop (matches reference)
    # ------------------------------------------------------------------
    def _read_pending_text(self) -> Optional[str]:
        next_idx, frame = self.text_input_buffer.read(self._next_text_index, 1)
        self._next_text_index = next_idx
        if frame is None:
            return None
        return decode_text_frame(frame[0])

    def is_start(self): return int(self.ctrl_buffer.recv()) == self._CMD_START
    def is_terminate(self): return int(self.ctrl_buffer.recv()) == self._CMD_TERMINATE
    def is_reset(self): return int(self.ctrl_buffer.recv()) == self._CMD_RESET

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
                    logger.info("streaming worker handling prompt: %r", user_text[:80])
                    self._handle_prompt(user_text)
                else:
                    time.sleep(0.001)
        except KeyboardInterrupt:
            self.terminate()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run the streaming Qwen3-Omni worker.")
    ap.add_argument("--cfg", required=True, help="streaming worker config YAML")
    StreamingWorker(cfg_path=ap.parse_args().cfg).loop()


if __name__ == "__main__":
    main()
