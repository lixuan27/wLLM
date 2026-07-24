"""VAE-group server for the disaggregated (DiT|VAE cross-chunk pipeline) variant.

The DiT group (server.py, role="dit") emits clean latents into a shm ring and
signals lifecycle via a shm control buffer; this VAE group consumes latents,
decodes them, and writes the video buffer — running concurrently with the DiT
group so VAE decode[N] overlaps DiT compute[N+1] (IR: Stage1 ∥ Stage2).

Intra-VAE multi-GPU (tile, Vr in {2,3,4}) uses the shared VAE-world collective
(the decoder's split_tile/gather_tile keyed off get_world_size()); the
DiT↔VAE handoff uses the repo's shm channel (the same IPC family the adapter
and ASR sidecar use), not a hand-rolled collective.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.logger import init_logger
from wllm.serving.rt_config import RTConfig
from wllm.serving.runner.vae_runner import VAERunner
from wllm.serving.utils.dtype import parse_dtype_getattr
from wllm.serving.distributed.parallel_state import (
    get_world_group, get_local_torch_device,
    maybe_init_distributed_environment_and_model_parallel,
)
from wllm.apps.longlive.backend.cuda import generation as G

logger = init_logger(__name__)
_LATENT_RING = 8       # must match server.py (_LATENT_RING): DiT->VAE latent ring depth


class _VaeCore:
    """Minimal core exposing what generation.decode_latent_frame needs."""
    def __init__(self, cfg, device, dtype, vae_runner):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype
        self.vae_runner = vae_runner
        self.ring = {"latent_decode_count": 0}


class VaeServer:
    def __init__(self, cfg: RTConfig, opts: dict):
        self.cfg = cfg
        self.world = int(os.environ.get("WORLD_SIZE", "1"))   # Vr
        self.rank = int(os.environ.get("RANK", "0"))
        self.is_rank0 = (self.rank == 0)
        if self.world > 1:
            maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
            self.device = get_local_torch_device()
        else:
            self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.dtype = parse_dtype_getattr(cfg.dtype)
        self.vae_runner = VAERunner(cfg, self.dtype, self.device)
        self.vae_runner.vae.decoder.world_size = self.world   # 1=local, 2..4=tile
        self.core = _VaeCore(cfg, self.device, self.dtype, self.vae_runner)
        self._warmup()

        self._lat_name = (cfg.video_buffer_name or "ll") + "_lat"
        self._vcmd_name = (cfg.video_buffer_name or "ll") + "_vcmd"
        self.readpos = 0
        self._last_cmd = 0
        self._last_seq = -1          # last prompt seq the VAE has tagged a video frame for
        if self.is_rank0:
            self.video_buffer = SharedTensorBuffer(
                cfg.video_buffer_name, frame_shape=(cfg.height, cfg.width, 3),
                max_len=cfg.max_num_frames, dtype=np.uint8, create=True)
            self.latent_buf = SharedTensorBuffer(
                self._lat_name,
                frame_shape=(cfg.dit_config.out_channels, int(cfg.chunk_size),
                             cfg.latent_height, cfg.latent_width),
                dtype=np.float32, max_len=_LATENT_RING, create=False, wait=True, timeout_s=1800.0)
            # latent tags from the DiT group: (prompt_seq, apply_ms) per latent chunk ->
            # first-video-frame-of-prompt tags + ASR-less latency for the harness.
            self.latent_tag = SharedTensorBuffer(
                self._lat_name + "_tag", frame_shape=(2,), dtype=np.int64,
                max_len=_LATENT_RING, create=False, wait=True, timeout_s=1800.0)
            # we bump this with our consumed-chunk index so the DiT can backpressure on it.
            self.vae_prog = SharedTensorBuffer(
                self._lat_name + "_prog", frame_shape=(1,), dtype=np.int64,
                max_len=1, create=False, wait=True, timeout_s=1800.0)
            self.tag_buffer = SharedTensorBuffer(
                cfg.video_buffer_name + "_tag", frame_shape=(3,), dtype=np.int64,
                max_len=512, create=True)
            self.vae_cmd = SharedControlBuffer(self._vcmd_name, create=False, wait=True,
                                               timeout_s=1800.0)
        logger.info("VAE group up rank=%d world=%d device=%s", self.rank, self.world, self.device)

    def _warmup(self):
        dummy = torch.zeros(1, self.cfg.vae_config.z_dim, 1, self.cfg.latent_height,
                            self.cfg.latent_width, device=self.device, dtype=self.dtype)
        self.vae_runner.run(dummy, True)
        self.vae_runner.run(dummy, False)
        self.vae_runner.clear()

    def _latents_from_np(self, np_lat):
        t = torch.from_numpy(np.ascontiguousarray(np_lat)).to(self.device, self.dtype)
        return t.unsqueeze(0)   # [1, z, chunk, lh, lw]

    def _decode_and_write(self, latents, seq=None, apply_ms=0):
        for l in range(int(self.cfg.chunk_size)):
            is_first = (self.core.ring["latent_decode_count"] == 0)
            frame = G.decode_latent_frame(self.core, latents, l, is_first)
            self.core.ring["latent_decode_count"] += 1
            if self.is_rank0:
                # first video frame of a new prompt -> tag (seq, frame_idx, asr_less_ms);
                # asr_less = apply(DiT) -> this frame(VAE), i.e. the full deploy path.
                if l == 0 and seq is not None and seq != self._last_seq:
                    asr_less = (int(time.time() * 1000) - int(apply_ms)) if apply_ms else 0
                    self.tag_buffer.write(np.array([seq, self.video_buffer.num, asr_less],
                                                   dtype=np.int64))
                    self._last_seq = seq
                arr = frame.detach().cpu().numpy() if torch.is_tensor(frame) else np.asarray(frame)
                self.video_buffer.write(arr)

    def _reset_state(self):
        self.core.ring["latent_decode_count"] = 0
        self.vae_runner.clear()
        self._last_seq = -1
        if self.is_rank0:
            self.video_buffer.clear()
            self.readpos = self.latent_buf.num
            self.vae_prog.write(np.array([self.readpos], dtype=np.int64))   # progress = caught up
            if getattr(self, "tag_buffer", None) is not None:
                self.tag_buffer.clear()

    # ------------------------------------------------------------------ loops
    def run(self):
        try:
            if self.world > 1:
                self._run_tile()
            else:
                self._run_single()
        finally:
            torch.cuda.empty_cache()

    def _run_single(self):
        while True:
            v = int(self.vae_cmd.recv())
            if v != self._last_cmd:
                op = v & 3
                self._last_cmd = v
                if op == 2:
                    self._teardown(); self.vae_cmd.commit(); break
                if op == 3:
                    self._reset_state(); self.vae_cmd.commit(); continue
            self.readpos, lat = self.latent_buf.read(self.readpos, 1)
            if lat is None:
                time.sleep(0.002); continue
            self.vae_prog.write(np.array([self.readpos], dtype=np.int64))   # consumed -> frees the ring slot
            _, tg = self.latent_tag.read(self.readpos - 1, 1)
            seq = int(tg[0][0]) if tg is not None else None
            apply_ms = int(tg[0][1]) if tg is not None else 0
            self._decode_and_write(self._latents_from_np(lat[0]), seq, apply_ms)

    def _run_tile(self):
        if self.is_rank0:
            while True:
                v = int(self.vae_cmd.recv())
                if v != self._last_cmd:
                    op = v & 3
                    self._last_cmd = v
                    get_world_group().broadcast_object({"op": op}, src=0)
                    if op == 2:
                        self._teardown(); self.vae_cmd.commit(); break
                    if op == 3:
                        self._reset_state(); self.vae_cmd.commit(); continue
                self.readpos, lat = self.latent_buf.read(self.readpos, 1)
                if lat is None:
                    get_world_group().broadcast_object({"op": 0}, src=0)
                    time.sleep(0.002); continue
                self.vae_prog.write(np.array([self.readpos], dtype=np.int64))   # consumed
                _, tg = self.latent_tag.read(self.readpos - 1, 1)
                seq = int(tg[0][0]) if tg is not None else None
                apply_ms = int(tg[0][1]) if tg is not None else 0
                get_world_group().broadcast_object({"op": 1}, src=0)
                latents = self._latents_from_np(lat[0])
                get_world_group().broadcast(latents, src=0)
                self._decode_and_write(latents, seq, apply_ms)
        else:
            shape = (1, self.cfg.dit_config.out_channels, int(self.cfg.chunk_size),
                     self.cfg.latent_height, self.cfg.latent_width)
            while True:
                cmd = get_world_group().broadcast_object(None, src=0)
                op = cmd["op"]
                if op == 2:
                    break
                if op == 3:
                    self._reset_state(); continue
                if op == 0:
                    continue
                latents = torch.empty(shape, device=self.device, dtype=self.dtype)
                get_world_group().broadcast(latents, src=0)
                self._decode_and_write(latents)

    def _teardown(self):
        if self.is_rank0:
            for b in (getattr(self, "video_buffer", None), getattr(self, "tag_buffer", None)):
                if b is not None:
                    try:
                        b.unlink()
                    except Exception:
                        pass


def main_vae():
    import json
    from wllm.serving.utils.torch_utils import set_torch_options
    set_torch_options()
    cfg = RTConfig.from_yaml(os.environ["CONFIG_PATH"], is_path=True)
    opts = json.loads(os.environ.get("LL_OPTS", "{}"))
    VaeServer(cfg, opts).run()


if __name__ == "__main__":
    main_vae()
