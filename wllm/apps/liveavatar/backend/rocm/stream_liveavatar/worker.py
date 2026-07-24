"""Variant `stream_liveavatar` — isolates the streaming-write lever (layer 1).

The reference worker computes the ENTIRE response (ASR -> LLM -> full TTS ->
all LiveAvatar chunks) and only writes video+audio to the output buffers at the
very end (worker.inference: one write of video[:n]/audio[:n]). So the first
output frame is not visible to the consumer until the whole response has been
generated -> latency-to-first-output = full pipeline time.

This variant changes ONLY the write schedule: `_run_liveavatar_reference`
writes each 480 ms chunk's 24 video frames + 24 audio frames to the output
buffers immediately as the chunk is produced, and returns (None, None) so the
base `inference()` skips its final batch write. Every other computation (ASR,
LLM, TTS, the per-chunk DiT+VAE) is byte-for-byte identical to the reference,
so the produced frames are identical — only *when* they appear changes.

IR basis: worker graph, TTS->LiveAvatar streaming edge + the per-chunk LiveAvatar
loop; the reference materializes the whole `liveavatar` stage before emitting,
this emits per chunk. Attacks: latency-to-first-output (and exposes the true
per-chunk streaming production fps for the sustainable-rate metric).

Single GPU for the DiT (same placement as the reference) — this isolates the
scheduling lever from any model parallelism.
"""
from __future__ import annotations

import numpy as np

from wllm.apps.liveavatar.reference.worker import LiveAvatarWorker


class StreamLiveAvatarWorker(LiveAvatarWorker):
    def _run_liveavatar_reference(self, audio_samples: np.ndarray):
        frames_per_latent = int(self.cfg.vae_config.scale_factor_temporal)
        step_frames = int(self.cfg.chunk_size) * frames_per_latent
        step_samples = int(self.cfg.tts_chunk_size)

        sample_offset = 0
        ran_step = False
        while True:
            if ran_step and sample_offset >= audio_samples.size:
                break

            chunk_audio = audio_samples[sample_offset:sample_offset + step_samples]
            actual = int(chunk_audio.size)
            if actual == 0 and ran_step:
                break
            if actual < step_samples:
                chunk_audio = np.pad(chunk_audio, (0, step_samples - actual), mode="constant")

            audio_features = self.extract_audio_features(chunk_audio, target_frames=step_frames)
            video_chunk = self.pipe.step(audio_input=audio_features)          # (24, H, W, 3) uint8
            audio_frames = self._frame_audio(np.asarray(chunk_audio, dtype=np.float32))  # (24, 320)

            # emit this chunk immediately, keeping video/audio 1:1 aligned
            n = min(video_chunk.shape[0], audio_frames.shape[0])
            if n > 0:
                self.pipe._video_buffer.write(video_chunk[:n])
                self.audio_output_buffer.write(audio_frames[:n])

            sample_offset += step_samples
            ran_step = True

        # already written incrementally; tell inference() there is nothing to flush
        return None, None
