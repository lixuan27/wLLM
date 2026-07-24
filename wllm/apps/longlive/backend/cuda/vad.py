"""Streaming VAD segmenter — vendored verbatim from
``wllm/apps/longlive/reference/worker.py`` (StreamingVADSegmenter / AudioState).

Vendored (not imported) so the deployment variants preserve byte-identical
audio->utterance segmentation behavior independent of the reference worker,
per the repo AGENTS.md "vendor the file into your workspace" rule. The logic
is unchanged; only the surrounding module differs.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


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
