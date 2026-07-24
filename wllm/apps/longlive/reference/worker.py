from __future__ import annotations

import os
import time
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.logger import init_logger
from wllm.apps.longlive.reference.config import LongLiveReferenceConfig
from wllm.apps.longlive.reference.pipeline import LongLivePipeline
from wllm.serving.utils.rand import set_global_seed
from wllm.serving.utils.torch_utils import set_torch_options

logger = init_logger(__name__)
set_torch_options()


class AudioState(Enum):
    IN_SILENCE = 0
    IN_SPEECH = 1


class StreamingVADSegmenter:
    def __init__(
        self,
        speech_threshold: float = 0.005,
        speech_start_frames: int = 10,
        speech_end_frames: int = 50,
        preroll_size: int = 5,
        min_utterance_frames: int = 25,
        max_utterance_frames: int = 2000,
        drop_trailing_silence: bool = True,
    ):
        self.speech_threshold = speech_threshold
        self.speech_start_frames = speech_start_frames
        self.speech_end_frames = speech_end_frames
        self.preroll_size = preroll_size
        self.min_utterance_frames = min_utterance_frames
        self.max_utterance_frames = max_utterance_frames
        self.drop_trailing_silence = drop_trailing_silence

        self.state = AudioState.IN_SILENCE
        self.silence_buffer: List[bool] = []
        self.speech_buffer: List[bool] = []
        self.preroll_frames: List[np.ndarray] = []
        self.utterance_frames: List[np.ndarray] = []

    def process_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        is_speech = rms >= self.speech_threshold

        self.preroll_frames.append(audio_chunk.copy())
        if len(self.preroll_frames) > self.preroll_size:
            self.preroll_frames.pop(0)

        if self.state == AudioState.IN_SILENCE:
            self.speech_buffer.append(is_speech)
            if len(self.speech_buffer) > self.speech_start_frames:
                self.speech_buffer.pop(0)

            self.silence_buffer.clear()
            if len(self.speech_buffer) == self.speech_start_frames and all(self.speech_buffer):
                self.state = AudioState.IN_SPEECH
                self.utterance_frames = [f.copy() for f in self.preroll_frames]
                self.speech_buffer.clear()
                self.silence_buffer.clear()

            return False, None

        self.utterance_frames.append(audio_chunk.copy())
        if len(self.utterance_frames) >= self.max_utterance_frames:
            audio = np.concatenate(self.utterance_frames, axis=0)
            self._reset_to_silence()
            return True, audio

        self.silence_buffer.append(not is_speech)
        if len(self.silence_buffer) > self.speech_end_frames:
            self.silence_buffer.pop(0)

        self.speech_buffer.clear()
        if len(self.silence_buffer) == self.speech_end_frames and all(self.silence_buffer):
            frames = self.utterance_frames
            if self.drop_trailing_silence and len(frames) >= self.speech_end_frames:
                frames = frames[:-self.speech_end_frames]

            if len(frames) < self.min_utterance_frames:
                self._reset_to_silence()
                return False, None

            audio = np.concatenate(frames, axis=0) if frames else None
            self._reset_to_silence()
            return audio is not None, audio

        return False, None

    def _reset_to_silence(self):
        self.state = AudioState.IN_SILENCE
        self.silence_buffer.clear()
        self.speech_buffer.clear()
        self.utterance_frames.clear()


class LongLiveWorker:
    """Sequential single-GPU reference worker for the LongLive app."""

    # Cap audio drain per loop iteration so a backlog can't starve the DiT.
    _MAX_AUDIO_CHUNKS_PER_TICK = 200

    def __init__(self, cfg_path: str):
        self.reference_cfg = LongLiveReferenceConfig.from_yaml(cfg_path, is_path=True)
        self.cfg = self.reference_cfg.to_runtime_config()
        self._init_worker()
        self.warmup()
        logger.info("LongLive backend READY")

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _create_pipeline(self):
        return LongLivePipeline(cfg=self.cfg, device=self.device)

    def _init_worker(self):
        if self.cfg.device != "cuda":
            raise ValueError(
                f"The reference worker expects cfg.device='cuda', got {self.cfg.device!r}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("The reference worker requires a visible CUDA GPU.")

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        self.pipe = self._create_pipeline()
        self.pipe.start_instance()

        self.session_started = False
        # Set once the first ASR transcript has been applied; clears on
        # reset() so the next session waits for its own first prompt.
        self.has_prompt = False

        # Control channel (start / terminate / reset opcodes).
        self.ctrl_buffer = SharedControlBuffer(
            self.cfg.ctrl_buffer_name, create=True,
        )

        # Microphone audio input buffer (the frontend writes 320-sample,
        # 16 kHz chunks here; the VAD + ASR turn them into prompt updates).
        self.audio_input_buffer = SharedTensorBuffer(
            self.cfg.audio_buffer_name,
            frame_shape=(int(self.cfg.audio_frame_samples),),
            dtype=np.float32,
            max_len=int(self.cfg.audio_max_chunks),
            create=True,
        )

        if self.cfg.signal_buffer_name is not None:
            self.signal_buffer = SharedControlBuffer(
                self.cfg.signal_buffer_name, create=True,
            )
        else:
            self.signal_buffer = None

        self.stream_segmenter = StreamingVADSegmenter()
        self.num_read_input_chunks = 0

        self._init_asr()

    def _init_asr(self):
        self.asr_model = Qwen3ASRModel.from_pretrained(
            self.cfg.asr_model_name,
            dtype=torch.bfloat16,
            device_map=str(self.device),
            attn_implementation="flash_attention_2",
            max_inference_batch_size=1,
            max_new_tokens=256,
        )
        logger.info("Loaded ASR model from %s on %s", self.cfg.asr_model_name, self.device)

        # Warm up ASR so the first user utterance doesn't pay the load cost.
        if os.getenv("WLLM_SKIP_ASR_WARMUP", "0") != "1":
            warmup_audio = np.zeros(
                (int(self.cfg.audio_sample_rate * 0.5),), dtype=np.float32,
            )
            try:
                self.asr_model.transcribe(
                    audio=(warmup_audio, int(self.cfg.audio_sample_rate)),
                    language="English",
                )
                logger.info("ASR warmup completed")
            except Exception as exc:
                logger.warning("ASR warmup skipped due to runtime error: %s", exc)

    # ------------------------------------------------------------------
    # session lifecycle
    # ------------------------------------------------------------------

    def warmup(self):
        set_global_seed(self.cfg.seed)
        warmup_prompt = self.cfg.prompt or "warmup"
        self.pipe.init_session(prompt=warmup_prompt)
        ring_capacity = max(
            int(self.cfg.chunk_size),
            int(self.cfg.context_window_size) + int(self.cfg.chunk_size),
        )
        warmup_blocks = (ring_capacity // int(self.cfg.chunk_size)) + 1
        for _ in range(warmup_blocks):
            self.pipe.step()
        self.pipe.reset()

    def start(self):
        set_global_seed(self.cfg.seed)
        self.session_started = True
        self.has_prompt = False
        self.pipe.reset()
        self.stream_segmenter = StreamingVADSegmenter()
        self.num_read_input_chunks = 0
        self.audio_input_buffer.clear()
        self.ctrl_buffer.commit()
        logger.info("LongLive session started (waiting for first prompt)")

    def reset(self):
        self.session_started = False
        self.has_prompt = False
        self.pipe.reset()
        self.stream_segmenter = StreamingVADSegmenter()
        self.num_read_input_chunks = 0
        self.audio_input_buffer.clear()
        self.ctrl_buffer.commit()
        logger.info("LongLive session reset")

    def terminate(self):
        # Idempotent: loop() calls this on a graceful ctrl-buffer TERM, and the
        # launcher's finally calls it on Ctrl+C / exit -- both must be safe.
        if getattr(self, "_terminated", False):
            return
        self._terminated = True
        self.session_started = False
        self.pipe.terminate_instance()
        self.ctrl_buffer.unlink()
        self.audio_input_buffer.unlink()
        if self.signal_buffer is not None:
            self.signal_buffer.unlink()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # prompt + asr
    # ------------------------------------------------------------------

    def _apply_prompt(self, text: str, is_first: bool) -> None:
        if is_first:
            self.pipe.init_session(prompt=text)
        else:
            self.pipe.update_prompt(prompt=text)
        self.has_prompt = True

    def _drain_audio_for_utterance(self) -> Optional[str]:
        utterance_audio: Optional[np.ndarray] = None
        for _ in range(self._MAX_AUDIO_CHUNKS_PER_TICK):
            self.num_read_input_chunks, audio_chunk = self.audio_input_buffer.read(
                x=self.num_read_input_chunks, n=1,
            )
            if audio_chunk is None:
                break
            should_infer, segmented = self.stream_segmenter.process_chunk(audio_chunk)
            if should_infer and segmented is not None:
                utterance_audio = segmented

        if utterance_audio is None:
            return None

        try:
            results = self.asr_model.transcribe(
                audio=(
                    np.asarray(utterance_audio, dtype=np.float32).reshape(-1),
                    int(self.cfg.audio_sample_rate),
                ),
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

    # ------------------------------------------------------------------
    # control buffer + main loop
    # ------------------------------------------------------------------

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

            new_text = self._drain_audio_for_utterance()
            if new_text is not None:
                is_first = not self.has_prompt
                self._apply_prompt(new_text, is_first=is_first)

            if self.has_prompt:
                self.pipe.step()
            else:
                time.sleep(0.005)
