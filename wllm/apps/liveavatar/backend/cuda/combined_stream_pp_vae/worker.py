"""combined_stream_pp_vae worker: app streaming + 5-stage (4 DiT + 1 VAE) pipeline.

Identical APP behavior to combined_stream_pp (stream TTS windows into the cluster,
emit frames as soon as the pipeline produces them), but the cluster includes the
VAE as a dedicated 5th rank, so this worker does NOT VAE-decode. It relays
already-decoded frames from the cluster's video queue into the adapter output
buffer. Net effect vs combined_stream_pp:
  - the worker GPU is freed of ALL VAE compute (only ASR + Wav2Vec remain), and
  - per-chunk latents move rank3 -> rank4 over NVLink p2p instead of the
    GPU->host->GPU round trip the worker did to decode them itself.

The generation (4 DiT steps) and the VAEDecode op are byte-for-byte the pp_steps /
combined_stream_pp ones, so the produced video is bit-exact with the reference by
the same construction. Only *where* the VAE op runs changed.

Needs GPU-to-GPU P2P (NVLink-class nodes) for the VAE to live inside the NCCL
world; on PCIe-only machines use combined_stream_pp, which keeps the VAE on the
worker GPU.

Layout: worker/ASR+Wav2Vec on device 0; DiT steps on 1-4; VAE on 5; LLM and TTS
on 6 (shared) or 6 and 7 (split). Launch via:

    python -m wllm.apps.liveavatar.backend.cuda.launch --variant combined_stream_pp_vae
    python -m wllm.apps.liveavatar.backend.cuda.launch --variant combined_stream_pp_vae_split
"""
import os
import sys
import time

import numpy as np


from wllm.apps.liveavatar.reference.pipeline import LiveAvatarPipeline
from wllm.apps.liveavatar.backend.cuda.pp_steps.worker import _RemotePipe
from wllm.apps.liveavatar.backend.cuda.combined_stream_pp.worker import CombinedStreamPPWorker
from wllm.apps.liveavatar.backend.cuda.combined_stream_pp_vae.cluster_client import ClusterClientVAE
from wllm.serving.logger import init_logger

logger = init_logger(__name__)


class _RemotePipeVAE(_RemotePipe):
    """Like pp_steps._RemotePipe, but the DiT cluster ALSO hosts the VAE rank, so
    this proxy never decodes. It hosts a local LiveAvatarPipeline only to OWN the
    adapter video buffer (cfg.video_buffer_name) and prime the session image; the
    decoded frames arrive pre-rendered on the cluster's video queue.

    (The local pipeline's DiT/VAE weights load but never run on the worker GPU --
    same as _RemotePipe's local DiT. A future cleanup could create the video buffer
    without loading the models; kept as-is here to reuse the proven path.)"""

    def __init__(self, cfg, device, cfg_path):
        super().__init__(cfg, device, cfg_path)
        self.prefix = f"ppwvae_{cfg.ctrl_buffer_name}"   # distinct shm namespace
        # 5 cluster gpus: 4 DiT steps + 1 VAE (override _RemotePipe's 4-gpu default).
        # Default fits the 8-GPU layout: worker g0 (ASR+Wav2Vec), cluster g1-5,
        # LLM g6, TTS g7. Override via PP_CLUSTER_GPUS; the launcher places
        # LLM/TTS through the derived config's llm/tts_gpu_index.
        self.cluster_gpus = [int(x) for x in
                             os.environ.get("PP_CLUSTER_GPUS", "1,2,3,4,5").split(",")]

    def start_instance(self):
        # local pipe OWNS the adapter video buffer (and primes the image); it does
        # NOT decode in this variant.
        self.local = LiveAvatarPipeline(cfg=self.cfg, device=self.device)
        self.local.start_instance()
        self._video_buffer = self.local._video_buffer
        # remote 5-rank cluster (4 DiT denoising steps + 1 VAE decode rank)
        self.client = ClusterClientVAE(self.cfg_path, self.prefix, self.cluster_gpus)
        self.client.wait_ready()
        logger.info("combined_stream_pp_vae cluster ready on gpus %s", self.cluster_gpus)


class CombinedStreamPPVAEWorker(CombinedStreamPPWorker):
    """Streaming worker over the 5-stage cluster.

    Reuses CombinedStreamPPWorker's streaming `inference`/`_inference_combined`
    verbatim; only three things change because the VAE now lives on the cluster:
    the pipeline factory, the warmup drain (drain video not latents), and
    `_drain_video` (relay decoded frames instead of decoding latents locally)."""

    def __init__(self, cfg_path):
        self._vid_read = 0   # monotonic cursor into the cluster's video queue
        super().__init__(cfg_path)

    def _create_pipeline(self):
        return _RemotePipeVAE(self.cfg, self.device, self._cfg_path)

    def warmup(self):
        # warm the cluster DiT + VAE shapes (one-time eager cudnn) by pushing dummy
        # feature chunks and draining the decoded frames the VAE rank emits. Use
        # self.pipe.init_session() (not just client.init_session()): it also runs
        # the LOCAL session init, warming the worker's VAE ENCODER (the image encode
        # in _on_session_init) so the first real request does not cold-benchmark it.
        client = self.pipe.client
        dummy = np.zeros((25, 1024, self._step_frames), dtype=np.float32)
        nwarm = max(6, int(self.cfg.context_window_size) // int(self.cfg.chunk_size) + 2)
        self.pipe.init_session()
        for _ in range(nwarm):
            client.push_features(dummy)
        target = nwarm * 24                      # 24 video frames per chunk (3 latents x sf_t=4 x 2)
        got, t0 = 0, time.time()
        while got < target and time.time() - t0 < 1800:
            self._vid_read, vid = client.video_buf.read(self._vid_read, 1)
            if vid is not None:
                got += 1
            else:
                time.sleep(0.002)
        logger.info("combined_stream_pp_vae warmup drained %d/%d video frames", got, target)

    def _drain_video(self) -> int:
        """Relay already-decoded frames from the cluster's video queue into the
        adapter output buffer. (In combined_stream_pp this method VAE-decoded the
        chunk's latents on the worker GPU; here rank4 already did that.) Returns the
        number of FRAMES emitted."""
        n = 0
        while True:
            self._vid_read, vid = self.pipe.client.video_buf.read(self._vid_read, 1)
            if vid is None:
                break
            self.pipe._video_buffer.write(np.asarray(vid[0]))   # (H,W,3) uint8, one frame
            self._req_collected += 1
            n += 1
        return n
