"""Adapter-facing worker frontend shared by LongLive agent variants.

Implements the exact control protocol, audio-ingest/VAD, ASR, and session
lifecycle the reference worker exposes (so the unchanged frontend + shared
adapter drive it identically), and delegates the *generation* to overridable
hooks. Single-process variants (baseline, and the coordinator rank of the
multi-GPU variants) subclass this.

Contract parity with ``wllm/apps/longlive/reference/worker.py``:
  * ctrl opcodes 1=start, 2=terminate, 3=reset;
  * 320-sample / 16 kHz audio frames -> VAD -> ASR -> prompt update;
  * one video chunk generated per loop tick once a prompt exists;
  * ``video_buffer_name`` shm buffer of ``(H, W, 3)`` uint8 frames.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.logger import init_logger
from wllm.serving.rt_config import RTConfig
from wllm.serving.utils.rand import set_global_seed
from wllm.serving.utils.torch_utils import set_torch_options

from wllm.apps.longlive.backend.rocm.vad import StreamingVADSegmenter

logger = init_logger(__name__)

# Match the reference worker (wllm/apps/longlive/reference/worker.py): enable TF32
# matmul + cudnn.benchmark + the dynamo/alloc knobs. Without this the exact same
# DiT/VAE compute runs ~30% slower per chunk (measured baseline 14.5 vs 18.9 fps).
set_torch_options()


class LongLiveWorkerBase:
    _MAX_AUDIO_CHUNKS_PER_TICK = 200

    def __init__(self, cfg_path: str):
        self.cfg = RTConfig.from_yaml(cfg_path, is_path=True)
        if not torch.cuda.is_available():
            raise RuntimeError("worker requires a visible CUDA/HIP GPU")
        # Use the already-selected device: cuda:0 for single-GPU variants, or
        # cuda:local_rank when a distributed variant has already called
        # torch.cuda.set_device before constructing the coordinator worker.
        self.device = torch.device("cuda", torch.cuda.current_device())

        self.session_started = False
        self.has_prompt = False

        self._init_io()
        self._init_asr()
        self._init_gen()          # hook: build generation backend + video buffer
        self._warmup()            # hook
        logger.info("LongLive agent worker ready (%s)", type(self).__name__)

    # ------------------------------------------------------------------
    # IO / control
    # ------------------------------------------------------------------
    def _init_io(self):
        self.ctrl_buffer = SharedControlBuffer(self.cfg.ctrl_buffer_name, create=True)
        self.audio_input_buffer = SharedTensorBuffer(
            self.cfg.audio_buffer_name,
            frame_shape=(int(self.cfg.audio_frame_samples),),
            dtype=np.float32,
            max_len=int(self.cfg.audio_max_chunks),
            create=True,
        )
        if self.cfg.signal_buffer_name is not None:
            self.signal_buffer = SharedControlBuffer(self.cfg.signal_buffer_name, create=True)
        else:
            self.signal_buffer = None
        self.stream_segmenter = StreamingVADSegmenter()
        self.num_read_input_chunks = 0

    def _create_video_buffer(self):
        return SharedTensorBuffer(
            name=self.cfg.video_buffer_name,
            frame_shape=(self.cfg.height, self.cfg.width, 3),
            max_len=self.cfg.max_num_frames,
            dtype=np.uint8,
            create=True,
        )

    def _init_asr(self):
        from qwen_asr import Qwen3ASRModel
        asr_device = os.environ.get("LL_ASR_DEVICE", str(self.device))
        self.asr_model = Qwen3ASRModel.from_pretrained(
            self.cfg.asr_model_name,
            dtype=torch.bfloat16,
            device_map=asr_device,
            attn_implementation="flash_attention_2",
            max_inference_batch_size=1,
            max_new_tokens=256,
        )
        logger.info("Loaded ASR model from %s on %s", self.cfg.asr_model_name, asr_device)
        if os.getenv("WLLM_SKIP_ASR_WARMUP", "0") != "1":
            warmup_audio = np.zeros((int(self.cfg.audio_sample_rate * 0.5),), dtype=np.float32)
            try:
                self.asr_model.transcribe(
                    audio=(warmup_audio, int(self.cfg.audio_sample_rate)), language="English"
                )
                logger.info("ASR warmup completed")
            except Exception as exc:
                logger.warning("ASR warmup skipped: %s", exc)

    # ------------------------------------------------------------------
    # generation hooks (subclass)
    # ------------------------------------------------------------------
    def _init_gen(self):
        raise NotImplementedError

    def _warmup(self):
        raise NotImplementedError

    def _reset_gen(self):
        raise NotImplementedError

    def _apply_prompt(self, text: str, is_first: bool):
        raise NotImplementedError

    def _step(self):
        """Generate + emit one chunk of video."""
        raise NotImplementedError

    def _teardown(self):
        pass

    # ------------------------------------------------------------------
    # ASR / prompt
    # ------------------------------------------------------------------
    def _transcribe(self, utterance_audio: np.ndarray) -> Optional[str]:
        try:
            results = self.asr_model.transcribe(
                audio=(np.asarray(utterance_audio, dtype=np.float32).reshape(-1),
                       int(self.cfg.audio_sample_rate)),
                language="English",
            )
        except Exception:
            logger.exception("LongLive ASR failed")
            return None
        text = (results[0].text or "").strip()
        if not text:
            return None
        logger.info("Prompt update from ASR: %s", text)
        return text

    def _drain_audio_for_utterance(self) -> Optional[np.ndarray]:
        utterance_audio = None
        for _ in range(self._MAX_AUDIO_CHUNKS_PER_TICK):
            self.num_read_input_chunks, audio_chunk = self.audio_input_buffer.read(
                x=self.num_read_input_chunks, n=1,
            )
            if audio_chunk is None:
                break
            # Pass the raw (1, 320) read result, exactly as the reference does;
            # the segmenter flattens frames to the same sample stream either way.
            should_infer, segmented = self.stream_segmenter.process_chunk(audio_chunk)
            if should_infer and segmented is not None:
                utterance_audio = segmented
        return utterance_audio

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self):
        set_global_seed(self.cfg.seed)
        self.session_started = True
        self.has_prompt = False
        self._reset_gen()
        self.stream_segmenter = StreamingVADSegmenter()
        self.num_read_input_chunks = 0
        self.audio_input_buffer.clear()
        self.ctrl_buffer.commit()
        logger.info("LongLive session started (waiting for first prompt)")

    def reset(self):
        self.session_started = False
        self.has_prompt = False
        self._reset_gen()
        self.stream_segmenter = StreamingVADSegmenter()
        self.num_read_input_chunks = 0
        self.audio_input_buffer.clear()
        self.ctrl_buffer.commit()
        logger.info("LongLive session reset")

    def terminate(self):
        self.session_started = False
        self._teardown()
        self.ctrl_buffer.unlink()
        self.audio_input_buffer.unlink()
        if self.signal_buffer is not None:
            self.signal_buffer.unlink()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def is_start(self) -> bool:
        return int(self.ctrl_buffer.recv()) == 1

    def is_terminate(self) -> bool:
        return int(self.ctrl_buffer.recv()) == 2

    def is_reset(self) -> bool:
        return int(self.ctrl_buffer.recv()) == 3

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

            utterance = self._drain_audio_for_utterance()
            if utterance is not None:
                new_text = self._transcribe(utterance)
                if new_text is not None:
                    is_first = not self.has_prompt
                    self._apply_prompt(new_text, is_first=is_first)
                    self.has_prompt = True

            if self.has_prompt:
                self._step()
            else:
                time.sleep(0.005)
