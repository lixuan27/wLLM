"""VAE half of the disaggregated DiT || VAE pipeline (torchrun v ranks, world=v).

Reads finalized chunk latents from the shm latent buffer produced by the DiT
process, decodes each latent (tile-parallel across its own v-rank world for v
in {2,3,4}, full-frame for v=1), and writes frames into the video buffer the
DiT process owns. The VAE temporal causal cache is threaded through the latents
in order, so the pixels match the reference (which decodes the same latents in
the same order).

This process does not print the backend's READY marker: it reports in over a
shm flag and the DiT process, which drives the frontend-facing buffers, emits
the single marker once both halves are up.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    get_world_rank, get_world_size, get_local_torch_device,
)
from wllm.serving.distributed.communication_op import global_broadcast, global_barrier
from wllm.serving.logger import init_logger
from wllm.serving.runner.vae_runner import VAERunner
from wllm.apps.worldplay.reference.config import WorldPlayReferenceConfig
from wllm.serving.utils.dtype import parse_dtype_getattr
from wllm.serving.utils.torch_utils import set_torch_options

logger = init_logger(__name__)
set_torch_options()

# The DiT process creates the buffers this one attaches to, but it is still
# loading its own model when we start, so wait generously.
ATTACH_TIMEOUT_S = 1800.0


class ShmVAEWorker:
    def __init__(self, cfg_path: str, v_size: int, shm_prefix: str, stream: bool = True):
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
        self.rank = get_world_rank()
        self.world = get_world_size()
        self.shm_prefix = shm_prefix
        self.stream = stream   # per-latent write (default) vs batch per chunk
        self.device = get_local_torch_device()
        torch.cuda.set_device(self.device)

        self.reference_cfg = WorldPlayReferenceConfig.from_yaml(cfg_path, is_path=True)
        self.cfg = self.reference_cfg.to_runtime_config()
        self.dtype = parse_dtype_getattr(self.cfg.dtype)

        self.vae_runner = VAERunner(self.cfg, self.dtype, self.device)
        self._latent_cursor = 0
        self._global_latent_idx = 0
        self.session_started = False

        cs = int(self.cfg.chunk_size)
        self._chunk_buf = torch.zeros(
            1, self.cfg.dit_config.out_channels, cs,
            self.cfg.latent_height, self.cfg.latent_width,
            device=self.device, dtype=torch.float32,
        )
        self._ctrl_pkt = torch.zeros(2, dtype=torch.int64, device=self.device)

        if self.rank == 0:
            # the DiT process owns the video buffer's lifecycle (create / clear /
            # unlink); we just open it and write frames into it.
            self.video_buffer = SharedTensorBuffer(
                name=self.cfg.video_buffer_name,
                frame_shape=(self.cfg.height, self.cfg.width, 3),
                max_len=int(self.cfg.max_num_frames), dtype=np.uint8,
                create=False, timeout_s=ATTACH_TIMEOUT_S,
            )
            self.latent_buffer = SharedTensorBuffer(
                f"{self.shm_prefix}_latent",
                frame_shape=(self.cfg.dit_config.out_channels, cs,
                             self.cfg.latent_height, self.cfg.latent_width),
                dtype=np.float32, max_len=64, create=False, timeout_s=ATTACH_TIMEOUT_S,
            )
            self.vae_ctrl = SharedControlBuffer(f"{self.shm_prefix}_vaectrl",
                                                create=False, timeout_s=ATTACH_TIMEOUT_S)
            self.vae_ready = SharedTensorBuffer(
                f"{self.shm_prefix}_vaeready", frame_shape=(1,), dtype=np.int64,
                max_len=1, create=False, timeout_s=ATTACH_TIMEOUT_S,
            )

        self._warmup()
        global_barrier()
        if self.rank == 0:
            self.vae_ready.write(np.ones((1,), dtype=np.int64))
            logger.info("WorldPlay VAE process up (v=%d, stream=%s)", self.world, self.stream)

    def _warmup(self):
        dummy = [torch.zeros(1, self.cfg.vae_config.z_dim, 1,
                             self.cfg.latent_height, self.cfg.latent_width,
                             device=self.device, dtype=self.dtype) for _ in range(3)]
        for i, d in enumerate(dummy):
            self.vae_runner.run(d, (i == 0))
        self.vae_runner.clear()

    def _decode_chunk(self, chunk):
        # chunk: [1, z, chunk_size, h, w] fp32, identical on all ranks
        cs = int(self.cfg.chunk_size)
        batch = []
        for j in range(cs):
            latent_i = chunk[:, :, j:j + 1, :, :]
            is_first = (self._global_latent_idx == 0)
            video_i = self.vae_runner.run(latent_i, is_first)   # collective tile decode + gather
            if self.rank == 0:
                frames = video_i[0].cpu().numpy()
                if self.stream:
                    self.video_buffer.write(frames)   # emit as soon as decoded
                else:
                    batch.append(frames)
            self._global_latent_idx += 1
        if self.rank == 0 and not self.stream and batch:
            self.video_buffer.write(np.concatenate(batch, axis=0))   # one write per chunk

    def _start_session(self):
        # the DiT already cleared the video buffer (its reset runs before it
        # relays start to us), so we only reset our own cursors + VAE cache.
        self.session_started = True
        self._latent_cursor = 0
        self._global_latent_idx = 0
        self.vae_runner.clear()
        if self.rank == 0:
            self.vae_ctrl.commit()

    def _reset_session(self):
        self.session_started = False
        self._latent_cursor = 0
        self._global_latent_idx = 0
        self.vae_runner.clear()
        if self.rank == 0:
            self.vae_ctrl.commit()

    def loop(self):
        while True:
            op = 0
            has_chunk = 0
            chunk_np = None
            if self.rank == 0:
                c = int(self.vae_ctrl.recv())
                if c == 2:
                    op = 2
                elif c == 1 and not self.session_started:
                    op = 1
                elif c == 3 and self.session_started:
                    op = 3
                if op == 0 and self.session_started:
                    self._latent_cursor, chunk_np = self.latent_buffer.read(self._latent_cursor, 1)
                    if chunk_np is not None:
                        has_chunk = 1
                self._ctrl_pkt[0] = op
                self._ctrl_pkt[1] = has_chunk
            if self.world > 1:
                global_broadcast(self._ctrl_pkt, src=0)
            op = int(self._ctrl_pkt[0].item())
            has_chunk = int(self._ctrl_pkt[1].item())

            if op == 2:
                if self.rank == 0:
                    self.vae_ctrl.commit()   # video buffer is unlinked by the DiT owner
                global_barrier()
                break
            elif op == 1:
                self._start_session()
            elif op == 3:
                self._reset_session()

            if has_chunk:
                if self.rank == 0:
                    # read(cursor, 1) returns [1, z, chunk, h, w] already (batch
                    # of one frame) — matches _chunk_buf, no unsqueeze needed.
                    self._chunk_buf.copy_(torch.from_numpy(chunk_np).to(self.device))
                if self.world > 1:
                    global_broadcast(self._chunk_buf, src=0)
                self._decode_chunk(self._chunk_buf)
            elif self.rank == 0:
                time.sleep(0.005 if not self.session_started else 0.0005)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", required=True, help="app runtime config YAML")
    ap.add_argument("--tiles", type=int, required=True,
                    help="number of ranks the VAE decode tiles across")
    ap.add_argument("--shm-prefix", required=True,
                    help="name prefix for the buffers shared with the DiT process")
    ap.add_argument("--vae-mode", choices=("stream", "batch"), default="stream",
                    help="write frames per latent as decoded, or once per chunk")
    args = ap.parse_args()
    ShmVAEWorker(
        cfg_path=args.cfg, v_size=args.tiles, shm_prefix=args.shm_prefix,
        stream=(args.vae_mode == "stream"),
    ).loop()


if __name__ == "__main__":
    main()
