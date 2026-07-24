"""stream_app variant worker.

Lever: APP-PIPELINE STREAMING (IR worker-graph streaming overlaps
tts→liveavatar). The reference runs ASR → full LLM → full TTS → full LiveAvatar
generation strictly sequentially and writes ALL video+audio frames in a single
burst at the very end (so latency-to-first-output == the whole pipeline, ~65 s,
and output is one burst). This variant overlaps TTS with the LiveAvatar
generation and EMITS each chunk's frames as soon as they are produced.

Correctness-preserving by construction: the LiveAvatar generation windows the
*concatenated, per-chunk-resampled* TTS audio into the same 7680-sample windows
as the reference's `_run_liveavatar_reference`, runs the same `pipe.step` on the
same features, and frames the same (padded) audio. Only the *timing* of when a
window is processed (streamed, as soon as 7680 samples are available) and when
output is emitted (incrementally) changes — not the content. Single GPU for the
DiT, so steady-state fps is unchanged (~reference); the win is latency + a
smooth continuous stream instead of a burst.

Subclasses the reference worker to reuse ASR/LLM/TTS/Wav2Vec/VAD/adapter setup
unchanged; only `inference` is overridden.
"""
import os
import sys
import uuid

import numpy as np
import torch
import torchaudio


from wllm.apps.liveavatar.reference.worker import LiveAvatarWorker, truncate_to_last_sentence, time_stretch_audio
from wllm.serving.logger import init_logger

logger = init_logger(__name__)


class StreamAppWorker(LiveAvatarWorker):

    async def _tts_stream(self, content: str):
        """Async generator yielding resampled (16 kHz, optionally stretched)
        audio arrays as the TTS engine produces them. Per-chunk processing is
        identical to the reference `_generate_tts_audio` (so concatenating the
        yielded arrays reproduces the reference's full audio bit-for-bit)."""
        tts_params = {
            "task_type": ["CustomVoice"],
            "text": [content],
            "language": [self.cfg.tts_language],
            "speaker": [self.cfg.tts_voice],
            "instruct": [""],
            "max_new_tokens": [2048],
        }
        prompt_obj = {
            "prompt_token_ids": [1] * self._estimate_tts_prompt_len(tts_params),
            "additional_information": tts_params,
        }
        seen_audio_chunks = 0
        consumed_tts_samples = 0
        async for stage_output in self.tts_engine.generate(
            prompt_obj, request_id=str(uuid.uuid4()), output_modalities=["audio"],
        ):
            new_chunks, seen_audio_chunks, sample_rate = self._extract_multimodal_audio(
                stage_output, seen_audio_chunks, consumed_tts_samples,
            )
            if not new_chunks:
                continue
            for chunk_tensor in new_chunks:
                if hasattr(chunk_tensor, "float"):
                    audio_array = chunk_tensor.float().detach().cpu().numpy()
                else:
                    audio_array = np.asarray(chunk_tensor)
                if audio_array.ndim > 1:
                    audio_array = audio_array.squeeze()
                audio_array = np.asarray(audio_array, dtype=np.float32).reshape(-1)
                if audio_array.size == 0:
                    continue
                consumed_tts_samples += int(audio_array.size)
                if float(self.cfg.audio_stretch_ratio) != 1.0:
                    audio_array = time_stretch_audio(audio_array, float(self.cfg.audio_stretch_ratio))
                audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
                if sample_rate != int(self.cfg.audio_sample_rate):
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor, sample_rate, int(self.cfg.audio_sample_rate))
                resampled = audio_tensor.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
                if resampled.size > 0:
                    yield resampled

    def _emit_window(self, window: np.ndarray, step_frames: int):
        """Run one DiT chunk on a 7680-sample window and write video+audio
        frames to the output buffers immediately (incremental emission)."""
        feats = self.extract_audio_features(window, target_frames=step_frames)
        video = self.pipe.step(audio_input=feats)          # (24, H, W, 3)
        audio_frames = self._frame_audio(window)            # (24, 320)
        n = min(video.shape[0], audio_frames.shape[0])
        if n > 0:
            self.pipe._video_buffer.write(video[:n])
            self.audio_output_buffer.write(audio_frames[:n])

    async def _inference_streaming(self, audio):
        asr_results = self.asr_model.transcribe(audio=(audio[0].flatten(), audio[1]),
                                                language="English")
        text = asr_results[0].text
        logger.info("ASR output: %s", text)
        self.message_history.append({"role": "user", "content": text})

        content = await self._generate_llm_response()     # LLM is sub-second; not streamed
        logger.info("LLM output: %s", content)
        content = truncate_to_last_sentence(content)
        self.message_history.append({"role": "assistant", "content": content})

        step_samples = int(self.cfg.tts_chunk_size)
        frames_per_latent = int(self.cfg.vae_config.scale_factor_temporal)
        step_frames = int(self.cfg.chunk_size) * frames_per_latent

        buf = np.empty((0,), dtype=np.float32)
        produced_any = False
        async for resampled in self._tts_stream(content):
            buf = np.concatenate([buf, resampled]) if buf.size else resampled
            while buf.shape[0] >= step_samples:           # same 7680 windowing as reference
                window = buf[:step_samples]
                buf = buf[step_samples:]
                self._emit_window(window, step_frames)
                produced_any = True
        # final partial window, padded (matches reference's last padded chunk)
        if buf.shape[0] > 0:
            window = np.pad(buf, (0, step_samples - buf.shape[0]))
            self._emit_window(window, step_frames)
            produced_any = True
        if not produced_any:
            logger.warning("TTS produced no audio; skipping LiveAvatar generation")

    def inference(self, audio):
        try:
            self._async_runner.run(self._inference_streaming(audio))
        except Exception:
            logger.exception("stream_app streaming inference failed")
