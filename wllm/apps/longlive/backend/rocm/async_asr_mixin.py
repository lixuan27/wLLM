"""Async-ASR mixin: run VAD + ASR on a background thread so the generation loop
never stalls for transcription; prompts are applied between chunks.

Mixed into a `LongLiveWorkerBase` subclass (the `async_asr` single-GPU worker,
and the `pipeline_dit4_vae4_async_asr` coordinator). The host class must provide the
usual gen hooks (`_init_gen`, `_apply_prompt`, `_step`, `_reset_gen`, ...); this
mixin only replaces the control loop + session lifecycle to move ASR off the
critical path. See `async_asr/worker.py` (isolated lever) and the IR worker
graph (ASR black-box independent of video_gen until the prompt handoff).
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional


class AsyncASRMixin:
    def _init_gen(self):
        super()._init_gen()
        self._pending: "queue.Queue[tuple]" = queue.Queue()
        self._asr_lock = threading.Lock()
        self._epoch = 0
        self._stop_asr = threading.Event()
        self._asr_thread = threading.Thread(target=self._asr_loop, daemon=True)
        self._asr_thread.start()

    def _asr_loop(self):
        while not self._stop_asr.is_set():
            if not self.session_started:
                time.sleep(0.005)
                continue
            with self._asr_lock:
                epoch = self._epoch
                utterance = self._drain_audio_for_utterance()
            if utterance is None:
                time.sleep(0.002)
                continue
            text = self._transcribe(utterance)        # slow, off the gen path
            if text is not None:
                self._pending.put((epoch, text))

    def start(self):
        with self._asr_lock:
            self._epoch += 1
            self._drain_pending()
        super().start()

    def reset(self):
        with self._asr_lock:
            self._epoch += 1
            self._drain_pending()
        super().reset()

    def terminate(self):
        self._stop_asr.set()
        if self._asr_thread.is_alive():
            self._asr_thread.join(timeout=5)
        super().terminate()

    def _drain_pending(self):
        try:
            while True:
                self._pending.get_nowait()
        except queue.Empty:
            pass

    def _next_prompt(self) -> Optional[str]:
        try:
            epoch, text = self._pending.get_nowait()
        except queue.Empty:
            return None
        if epoch != self._epoch:
            return None
        return text

    def loop(self):
        while True:
            if self.is_terminate() and self.session_started:
                self.terminate()
                break
            elif self.is_start() and not self.session_started:
                self.start()
            elif self.is_reset() and self.session_started:
                self.reset()

            if not self.session_started:
                time.sleep(0.005)
                continue

            text = self._next_prompt()
            if text is not None:
                is_first = not self.has_prompt
                self._apply_prompt(text, is_first=is_first)
                self.has_prompt = True

            if self.has_prompt:
                self._step()
            else:
                time.sleep(0.002)
