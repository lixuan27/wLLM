"""Variant `wav2vec_offload` — wav2vec feature extraction off the DiT GPU (IR L6).

IR basis: below the IR granularity — wav2vec feature extraction is independent of
the DiT KV state (worker graph: it feeds the `liveavatar` stage's audio input but
touches none of its persistent state), so it can run on a separate GPU and be
PREFETCHED for chunk N+1 while the DiT decodes chunk N. Attacks: sustainable rate
(remove wav2vec from the per-chunk DiT critical path) — expected small (wav2vec is
light vs the 14B DiT), measured to confirm.

Builds on stream_liveavatar (streaming write). wav2vec runs on WAV2VEC_DEVICE
(default cuda:1); the streaming loop prefetches the next chunk's features on that
GPU while pipe.step() runs on cuda:0, then transfers the features over.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from wllm.apps.liveavatar.backend.rocm.stream_liveavatar.worker import StreamLiveAvatarWorker


class Wav2VecOffloadWorker(StreamLiveAvatarWorker):
    def __init__(self, cfg_path: str):
        super().__init__(cfg_path)
        self._w2v_device = torch.device(os.environ.get("WAV2VEC_DEVICE", "cuda:1"))
        self.wav2vec_model.to(self._w2v_device)

    def _extract_on_w2v(self, audio_samples: np.ndarray, target_frames: int) -> torch.Tensor:
        inputs = self.wav2vec_processor(audio_samples, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(device=self._w2v_device, dtype=torch.bfloat16)
        res = self.wav2vec_model(input_values, output_hidden_states=True)
        hs = res.hidden_states
        stacked = torch.cat([h.squeeze(0) for h in hs], dim=0)
        stacked = stacked.view(len(hs), hs[0].shape[1], hs[0].shape[-1]).permute(0, 2, 1)
        stacked = F.interpolate(stacked.float(), size=target_frames, mode="linear", align_corners=True)
        return stacked.unsqueeze(0).to(dtype=torch.bfloat16)  # stays on _w2v_device

    def _run_liveavatar_reference(self, audio_samples: np.ndarray):
        frames_per_latent = int(self.cfg.vae_config.scale_factor_temporal)
        step_frames = int(self.cfg.chunk_size) * frames_per_latent
        step_samples = int(self.cfg.tts_chunk_size)

        # collect the response's 480 ms chunks up front
        chunks = []
        off = 0
        while off < audio_samples.size or not chunks:
            ca = audio_samples[off:off + step_samples]
            if ca.size == 0 and chunks:
                break
            if ca.size < step_samples:
                ca = np.pad(ca, (0, step_samples - ca.size), mode="constant")
            chunks.append(ca)
            off += step_samples

        # prefetch chunk 0's features on the wav2vec GPU
        feat = self._extract_on_w2v(chunks[0], step_frames).to(self.device, non_blocking=True)
        for i in range(len(chunks)):
            # kick off the NEXT chunk's wav2vec on the offload GPU (overlaps pipe.step)
            next_feat = None
            if i + 1 < len(chunks):
                next_feat = self._extract_on_w2v(chunks[i + 1], step_frames)
            video_chunk = self.pipe.step(audio_input=feat)              # DiT+VAE on cuda:0
            audio_frames = self._frame_audio(np.asarray(chunks[i], dtype=np.float32))
            n = min(video_chunk.shape[0], audio_frames.shape[0])
            if n > 0:
                self.pipe._video_buffer.write(video_chunk[:n])
                self.audio_output_buffer.write(audio_frames[:n])
            if next_feat is not None:
                feat = next_feat.to(self.device, non_blocking=True)
        return None, None
