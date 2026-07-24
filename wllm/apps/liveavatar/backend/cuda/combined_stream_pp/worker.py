"""combined_stream_pp variant worker.

Stacks the two winning levers:
  - APP STREAMING (stream_app): overlap TTS with generation; emit frames as soon
    as produced -> low latency-to-first-output + smooth stream.
  - CROSS-CHUNK STEP PIPELINE (pp_steps): the 4 denoising steps on 4 GPUs ->
    high sustainable throughput (>= real-time fps).

IR basis: worker-graph streaming overlaps (tts->liveavatar) AND
find_pipeline_stages (dit_step_i || dit_step_j independent across chunks). This
is the "non-dominated winning pair -> obligate a combination" variant: stream_app
wins latency, pp_steps wins throughput; together they target real-time
interaction (low latency AND >=50 fps).

Reuses PPStepsWorker (the cluster proxy + warmup/start/reset/terminate) and
StreamAppWorker (_tts_stream). Only `inference` is new: it streams TTS windows
into the cluster as they arrive and drains decoded frames from the cluster's
video queue into the adapter's output buffer incrementally.
"""
import os
import sys
import time

import numpy as np


from wllm.apps.liveavatar.reference.worker import truncate_to_last_sentence
from wllm.apps.liveavatar.backend.cuda.pp_steps.worker import PPStepsWorker
from wllm.apps.liveavatar.backend.cuda.stream_app.worker import StreamAppWorker
from wllm.serving.logger import init_logger

logger = init_logger(__name__)


class CombinedStreamPPWorker(PPStepsWorker, StreamAppWorker):
    # MRO: Combined -> PPStepsWorker -> StreamAppWorker -> LiveAvatarWorker
    #   _create_pipeline / warmup / start / reset / terminate  <- PPStepsWorker
    #   _tts_stream                                            <- StreamAppWorker
    #   inference                                              <- here

    def _drain_video(self) -> int:
        """Decode any ready cluster latent-chunks (the cluster ships final latents,
        not video -- see pp_steps) into the adapter output buffer. The VAE decode
        runs here on the worker's GPU as soon as a chunk's latents arrive, so
        frames still stream out incrementally. Returns the number of FRAMES emitted."""
        n = 0
        while True:
            self._lat_read, lat = self.pipe.client.latents_buf.read(self._lat_read, 1)
            if lat is None:
                break
            frames = self.pipe.decode_latents(np.asarray(lat[0]))   # (24,H,W,3)
            self.pipe._video_buffer.write(frames)
            self._req_collected += int(frames.shape[0])
            n += int(frames.shape[0])
        return n

    async def _inference_combined(self, audio):
        asr_results = self.asr_model.transcribe(audio=(audio[0].flatten(), audio[1]),
                                                language="English")
        text = asr_results[0].text
        logger.info("ASR output: %s", text)
        self.message_history.append({"role": "user", "content": text})
        content = await self._generate_llm_response()
        logger.info("LLM output: %s", content)
        content = truncate_to_last_sentence(content)
        self.message_history.append({"role": "assistant", "content": content})

        step = int(self.cfg.tts_chunk_size)
        step_frames = self._step_frames
        buf = np.empty((0,), dtype=np.float32)
        pushed = 0
        self._req_collected = 0

        async for resampled in self._tts_stream(content):
            buf = np.concatenate([buf, resampled]) if buf.size else resampled
            while buf.shape[0] >= step:
                window = buf[:step]
                buf = buf[step:]
                feats = self.extract_audio_features(window, target_frames=step_frames)
                self.pipe.client.push_features(feats.squeeze(0).float().cpu().numpy())
                self.audio_output_buffer.write(self._frame_audio(window))
                pushed += 1
                self._drain_video()           # emit whatever the pipeline finished
        if buf.shape[0] > 0:
            window = np.pad(buf, (0, step - buf.shape[0]))
            feats = self.extract_audio_features(window, target_frames=step_frames)
            self.pipe.client.push_features(feats.squeeze(0).float().cpu().numpy())
            self.audio_output_buffer.write(self._frame_audio(window))
            pushed += 1

        # drain the pipeline tail (frames still in flight)
        target = pushed * 24
        t0 = time.time()
        while self._req_collected < target and time.time() - t0 < 1800.0:
            if self._drain_video() == 0:
                time.sleep(0.003)
        if self._req_collected < target:
            logger.warning("combined: collected %d/%d frames before timeout",
                           self._req_collected, target)

    def inference(self, audio):
        try:
            self._async_runner.run(self._inference_combined(audio))
        except Exception:
            logger.exception("combined_stream_pp inference failed")
