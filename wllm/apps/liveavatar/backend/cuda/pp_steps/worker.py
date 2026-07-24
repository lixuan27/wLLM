"""pp_steps variant worker.

Lever: cross-chunk PIPELINE of the 4 DiT denoising steps across GPUs (the IR
find_pipeline_stages / cross-chunk-independent-pairs result). This worker keeps
the reference's APP scheduling (ASR -> full LLM -> full TTS -> generate -> emit
at the end, a burst) so the ONLY change vs the reference is that the LiveAvatar
DiT generation runs on the 4-GPU pipeline cluster instead of one GPU -> isolates
the throughput lever. (Streaming is added separately in combined_stream_pp.)

The 4 DiT steps are hosted by the pp_steps cluster (pp_steps/cluster.py); the
cluster ships FINAL LATENTS (the VAE decode does NOT run inside the dist world --
the eager causal-VAE cudnn benchmark stalls a rank that is inside the NCCL world;
see cluster.shm_names). This worker is the reference worker's front half
(ASR/LLM/TTS/Wav2Vec/VAD/adapter) with the local DiT pipeline replaced by a
_RemotePipe proxy that ships per-chunk wav2vec features to the cluster, then
VAE-decodes the returned latents on the worker's own GPU (a plain, fast,
non-dist VAE -- identical op to the reference's pipeline.step decode).

GPU layout (env PP_CLUSTER_GPUS, default 1,2,3,4): worker (ASR+Wav2Vec+VAE) on
its CVD cuda:0; LLM+TTS on cfg llm/tts_gpu_index (global); cluster on the
PP_CLUSTER_GPUS.
"""
import os
import sys
import time
from types import SimpleNamespace

import numpy as np


import torch

from wllm.apps.liveavatar.reference.worker import LiveAvatarWorker, truncate_to_last_sentence
from wllm.apps.liveavatar.reference.pipeline import LiveAvatarPipeline
from wllm.serving.utils.rand import set_global_seed
from wllm.apps.liveavatar.backend.cuda.pp_steps.cluster_client import ClusterClient
from wllm.apps.liveavatar.backend.cuda.pp_steps.cluster import SimpleState
from wllm.apps.liveavatar.backend.cuda.ir.ops import VAEDecode
from wllm.serving.logger import init_logger

logger = init_logger(__name__)


class _RemotePipe:
    """LiveAvatarPipeline-interface proxy backed by the pp_steps DiT cluster.

    Hosts a LOCAL LiveAvatarPipeline purely for its (non-dist, fast) VAE decode
    and the session image-encode/prime + the adapter video buffer; the 4 DiT
    denoising steps run on the remote cluster. The local DiT weights load but
    never run (the cluster does the DiT), so no local DiT compile happens.

    The worker uses: start_instance, init_session, decode_latents, reset,
    terminate_instance, and _video_buffer (the adapter output buffer)."""

    def __init__(self, cfg, device, cfg_path):
        self.cfg = cfg
        self.device = device
        self.cfg_path = cfg_path
        self.client = None
        self.local = None          # local pipeline: VAE decode + image encode/prime
        self._video_buffer = None
        self._vae_op = VAEDecode()
        self.prefix = f"ppw_{cfg.ctrl_buffer_name}"
        self.cluster_gpus = [int(x) for x in
                             os.environ.get("PP_CLUSTER_GPUS", "1,2,3,4").split(",")]

    def start_instance(self):
        # local pipe hosts the fast non-dist VAE and OWNS the adapter video buffer
        self.local = LiveAvatarPipeline(cfg=self.cfg, device=self.device)
        self.local.start_instance()
        self._video_buffer = self.local._video_buffer
        # remote DiT cluster (4 ranks, one denoising step each)
        self.client = ClusterClient(self.cfg_path, self.prefix, self.cluster_gpus)
        self.client.wait_ready()
        logger.info("pp_steps cluster ready on gpus %s", self.cluster_gpus)

    def init_session(self, prompt=None, negative_prompt=None, image_path=None):
        # local: encode image, prime VAE decoder cache; cluster: seed + init all ranks
        self.local.init_session(prompt=self.cfg.prompt, negative_prompt=None,
                                image_path=self.cfg.image_path)
        self.client.init_session()

    @torch.inference_mode()
    def decode_latents(self, lat_np: np.ndarray) -> np.ndarray:
        """VAE-decode one chunk's final latents (C,gen,h,w) f32 -> (frames,H,W,3).

        Uses the exact Phase-2-validated VAEDecode op + the local VAE's temporal
        cache (continuous across chunks, mirroring pipeline.step). MUST run under
        inference_mode: without it the per-frame VAE decode builds and retains an
        autograd graph across the continuous multi-request session -> the worker
        GPU OOMs after a few hundred frames."""
        lat = torch.from_numpy(lat_np).unsqueeze(0).to(self.device, self.local.dtype)
        vctx = SimpleNamespace(vae_runner=self.local.vae_runner)
        out = self._vae_op.execute({"latents": lat}, vctx,
                                   SimpleState({"vae_cache": self.local.vae_runner}))
        return out["video"]

    def reset(self):
        if self._video_buffer is not None:
            self._video_buffer.clear()

    def terminate_instance(self):
        if self.client is not None:
            self.client.terminate()
        if self.local is not None:
            self.local.terminate_instance()


class PPStepsWorker(LiveAvatarWorker):
    def __init__(self, cfg_path):
        self._cfg_path = cfg_path
        self._lat_read = 0   # continuous read cursor into the cluster latents queue
        super().__init__(cfg_path)

    def _create_pipeline(self):
        return _RemotePipe(self.cfg, self.device, self._cfg_path)

    @property
    def _step_frames(self):
        return int(self.cfg.chunk_size) * int(self.cfg.vae_config.scale_factor_temporal)

    def _drain_latents(self, n, timeout, decode):
        """Read n latent chunks from the cluster; optionally VAE-decode each.
        Returns the list of decoded frame-arrays (empty if decode=False)."""
        got, frames = 0, []
        t0 = time.time()
        while got < n and time.time() - t0 < timeout:
            self._lat_read, lat = self.pipe.client.latents_buf.read(self._lat_read, 1)
            if lat is not None:
                got += 1
                if decode:
                    frames.append(self.pipe.decode_latents(np.asarray(lat[0])))
            else:
                time.sleep(0.002)
        return got, frames

    def warmup(self):
        # warm the cluster DiT shapes (one-time eager cudnn) by pushing dummy
        # feature chunks. Use self.pipe.init_session() (not just
        # client.init_session()): it also runs the LOCAL session init, which warms
        # the worker's VAE ENCODER (the image encode in _on_session_init) so the
        # first real request does not cold-benchmark it. (The local start_instance
        # only warms the VAE DECODER.)
        client = self.pipe.client
        dummy = np.zeros((25, 1024, self._step_frames), dtype=np.float32)
        nwarm = max(6, int(self.cfg.context_window_size) // int(self.cfg.chunk_size) + 2)
        self.pipe.init_session()
        for _ in range(nwarm):
            client.push_features(dummy)
        got, _ = self._drain_latents(nwarm, timeout=1800, decode=False)
        logger.info("pp_steps cluster warmup drained %d/%d latent chunks", got, nwarm)

    def start(self):
        set_global_seed(self.cfg.seed)
        self.session_started = True
        self.pipe.init_session()   # local VAE prime + seed/init on every cluster rank
        self.pipe.reset()
        # _lat_read is a monotonic cursor into the growing latents queue; the
        # warmup drain already advanced it past the warmup chunks, so leave it.
        self.audio_input_buffer.clear()
        self.audio_output_buffer.clear()
        self.ctrl_buffer.commit()
        logger.info("pp_steps session started")

    def reset(self):
        self.session_started = False
        self.pipe.client.reset_session()
        self.pipe.init_session()   # re-prime the local VAE cache for the new session
        self.pipe.reset()
        self.audio_input_buffer.clear()
        self.audio_output_buffer.clear()
        self.ctrl_buffer.commit()
        logger.info("pp_steps session reset")

    def _window_audio(self, audio_samples: np.ndarray):
        """Same 7680-sample windowing as reference _run_liveavatar_reference."""
        step = int(self.cfg.tts_chunk_size)
        windows = []
        offset = 0
        ran = False
        while True:
            if ran and offset >= audio_samples.size:
                break
            chunk = audio_samples[offset:offset + step]
            if chunk.size == 0 and ran:
                break
            if chunk.size < step:
                chunk = np.pad(chunk, (0, step - chunk.size), mode="constant")
            windows.append(np.asarray(chunk, dtype=np.float32))
            offset += step
            ran = True
        return windows

    def inference(self, audio):
        # ---- reference app scheduling: ASR -> full LLM -> full TTS ----
        asr_results = self.asr_model.transcribe(audio=(audio[0].flatten(), audio[1]),
                                                language="English")
        text = asr_results[0].text
        logger.info("ASR output: %s", text)
        self.message_history.append({"role": "user", "content": text})
        content = self._async_runner.run(self._generate_llm_response())
        logger.info("LLM output: %s", content)
        content = truncate_to_last_sentence(content)
        self.message_history.append({"role": "assistant", "content": content})
        audio_samples = self._async_runner.run(self._generate_tts_audio(content))
        if audio_samples.size == 0:
            logger.warning("TTS produced no audio; skipping LiveAvatar generation")
            return

        # ---- generate on the cluster, decode locally, emit as one burst ----
        windows = self._window_audio(audio_samples)
        for w in windows:
            feats = self.extract_audio_features(w, target_frames=self._step_frames)
            self.pipe.client.push_features(feats.squeeze(0).float().cpu().numpy())

        got, decoded = self._drain_latents(len(windows), timeout=1800, decode=True)
        video_frames = np.concatenate(decoded, axis=0) if decoded else np.zeros(
            (0, self.cfg.height, self.cfg.width, 3), dtype=np.uint8)

        audio_frames = self._frame_audio(np.concatenate(
            [w for w in windows]).astype(np.float32))
        n = min(video_frames.shape[0], audio_frames.shape[0])
        if n > 0:
            self.pipe._video_buffer.write(video_frames[:n])
            self.audio_output_buffer.write(audio_frames[:n])

    def terminate(self):
        # Idempotent: loop() calls this on a graceful ctrl-buffer TERM, and the
        # launcher's finally calls it on Ctrl+C / exit -- both must be safe.
        # pipe.terminate_instance() -> ClusterClient.terminate() killpg's the
        # setsid-detached DiT+VAE ranks (which the terminal's SIGINT never reaches).
        if getattr(self, "_terminated", False):
            return
        self._terminated = True
        self.session_started = False
        self.pipe.terminate_instance()
        if self.llm_engine is not None:
            self.llm_engine.shutdown()
        if self.tts_engine is not None:
            self.tts_engine.shutdown()
        if self._async_runner is not None:
            self._async_runner.close()
        self.ctrl_buffer.unlink()
        self.audio_input_buffer.unlink()
        self.audio_output_buffer.unlink()
        if self.signal_buffer is not None:
            self.signal_buffer.unlink()
