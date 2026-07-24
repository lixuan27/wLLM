"""Variant `stream_full` — overlap TTS with the LiveAvatar generation (IR L1).

IR basis: the worker-graph `tts -> liveavatar` streaming edge (VARIABLE_RATE). The
reference (and stream_liveavatar) wait for the ENTIRE TTS audio before starting the
LiveAvatar DiT; here the DiT starts on the FIRST 480 ms of TTS audio while TTS keeps
generating the rest. Because the TTS engine is a separate engine process, it
produces the remaining audio in the background while pipe.step() blocks cuda:0, so
TTS and the DiT overlap for free. Attacks latency-to-first-output (removes the
full-TTS wait) beyond stream_liveavatar.

Builds on stream_liveavatar's per-chunk streaming write; adds a streaming TTS
consumer. LLM stays blocking (full text first — 80 tokens is fast) to keep the
transcript-conditioned response identical to the reference (correctness).
"""
from __future__ import annotations

import uuid

import numpy as np
import torch
import torchaudio

from wllm.apps.liveavatar.backend.rocm.stream_liveavatar.worker import StreamLiveAvatarWorker
from wllm.apps.liveavatar.reference.worker import time_stretch_audio


class StreamFullWorker(StreamLiveAvatarWorker):
    async def _tts_audio_stream(self, content: str):
        """Same as the reference _generate_tts_audio but YIELDS each resampled
        chunk as it arrives instead of concatenating at the end."""
        tts_params = {
            "task_type": ["CustomVoice"], "text": [content],
            "language": [self.cfg.tts_language], "speaker": [self.cfg.tts_voice],
            "instruct": [""], "max_new_tokens": [2048],
        }
        prompt_obj = {
            "prompt_token_ids": [1] * self._estimate_tts_prompt_len(tts_params),
            "additional_information": tts_params,
        }
        seen, consumed = 0, 0
        async for stage_output in self.tts_engine.generate(
            prompt_obj, request_id=str(uuid.uuid4()), output_modalities=["audio"]):
            new_chunks, seen, sr = self._extract_multimodal_audio(stage_output, seen, consumed)
            if not new_chunks:
                continue
            for ct in new_chunks:
                a = ct.float().detach().cpu().numpy() if hasattr(ct, "float") else np.asarray(ct)
                if a.ndim > 1:
                    a = a.squeeze()
                a = np.asarray(a, dtype=np.float32).reshape(-1)
                if a.size == 0:
                    continue
                consumed += int(a.size)
                if float(self.cfg.audio_stretch_ratio) != 1.0:
                    a = time_stretch_audio(a, float(self.cfg.audio_stretch_ratio))
                t = torch.from_numpy(a).unsqueeze(0)
                if sr != int(self.cfg.audio_sample_rate):
                    t = torchaudio.functional.resample(t, sr, int(self.cfg.audio_sample_rate))
                r = t.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
                if r.size > 0:
                    yield r

    def _process_chunk(self, chunk_audio: np.ndarray, step_frames: int):
        feat = self.extract_audio_features(chunk_audio, target_frames=step_frames)
        video = self.pipe.step(audio_input=feat)
        audio_frames = self._frame_audio(np.asarray(chunk_audio, dtype=np.float32))
        n = min(video.shape[0], audio_frames.shape[0])
        if n > 0:
            self.pipe._video_buffer.write(video[:n])
            self.audio_output_buffer.write(audio_frames[:n])

    async def _stream_tts_to_liveavatar(self, content: str):
        step_samples = int(self.cfg.tts_chunk_size)
        step_frames = int(self.cfg.chunk_size) * int(self.cfg.vae_config.scale_factor_temporal)
        buf = np.empty((0,), dtype=np.float32)
        produced = False
        async for r in self._tts_audio_stream(content):
            buf = np.concatenate([buf, r])
            while buf.size >= step_samples:
                chunk = buf[:step_samples]
                buf = buf[step_samples:]
                self._process_chunk(chunk, step_frames)   # DiT+VAE on cuda:0; TTS runs ahead in its process
                produced = True
        if buf.size > 0:
            chunk = np.pad(buf, (0, step_samples - buf.size), mode="constant")
            self._process_chunk(chunk, step_frames)
            produced = True
        return produced

    def _run_liveavatar_reference(self, audio_samples: np.ndarray):
        # not used — stream_full drives the LiveAvatar directly from the TTS stream
        return None, None

    def inference(self, audio):
        asr_results = self.asr_model.transcribe(audio=(audio[0].flatten(), audio[1]), language="English")
        text = asr_results[0].text
        self.message_history.append({"role": "user", "content": text})
        content = self._async_runner.run(self._generate_llm_response())
        from wllm.apps.liveavatar.reference.worker import truncate_to_last_sentence
        content = truncate_to_last_sentence(content)
        self.message_history.append({"role": "assistant", "content": content})
        self._async_runner.run(self._stream_tts_to_liveavatar(content))
