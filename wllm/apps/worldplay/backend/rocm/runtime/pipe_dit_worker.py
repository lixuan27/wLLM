"""DiT half of the disaggregated DiT || VAE pipeline (torchrun d ranks, SP=d).

Reuses `ColocatedSPWorker` for dist init / camera decode / control loop, but:
  * runs `DiTOnlyPipeline` (no VAE decode),
  * writes each chunk's finalized fp32 latents to a shared-memory latent buffer
    for the VAE process to consume,
  * relays session control (start/reset/terminate) to the VAE process over a
    second control buffer.
The two processes run fully decoupled in the steady state, so VAE(N) overlaps
DiT(N+1).

This process is the backend's driver: it owns the frontend-facing buffers and
emits the single READY marker, gated on the VAE process reporting in.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np
import torch

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.distributed.communication_op import global_broadcast
from wllm.serving.distributed.parallel_state import get_world_rank
from wllm.serving.logger import init_logger

from wllm.apps.worldplay.backend.rocm.runtime.colocated_pipeline import ColocatedPipeline
from wllm.apps.worldplay.backend.rocm.runtime.colocated_worker import ColocatedSPWorker

logger = init_logger(__name__)

# How long the DiT process waits for the VAE process to finish loading before
# giving up. Both sides load a full model and compile, so this is generous.
VAE_READY_TIMEOUT_S = 1800.0


class DiTOnlyPipeline(ColocatedPipeline):
    """Run the full denoise but, instead of decoding to pixels, hand the
    finalized chunk latents back to the worker (which ships them over shm to
    the VAE process)."""

    @torch.inference_mode()
    def _emit_frames(self, start_idx: int, end_idx: int) -> Optional[torch.Tensor]:
        # rank 0 returns the finalized fp32 chunk latents [1, z, chunk, h, w];
        # no VAE decode happens on the DiT side.
        if get_world_rank() != 0:
            return None
        return self._latents[:, :, start_idx:end_idx, :, :].clone()


class ShmDiTWorker(ColocatedSPWorker):
    def __init__(self, cfg_path: str, sp_size: int, shm_prefix: str):
        self.shm_prefix = shm_prefix
        super().__init__(cfg_path=cfg_path, sp_size=sp_size, vae_tile=False, stream=False)

    def _make_pipeline(self):
        # The DiT process CREATES + owns the real video buffer (its shm handles
        # survive the whole run; a VAE-created buffer was getting prematurely
        # unlinked by the py3.13 resource_tracker). The DiT never writes to it —
        # DiTOnlyPipeline emits latents, not pixels — it only creates/clears (on
        # reset) / unlinks (on terminate) it; the VAE process opens it and writes.
        return DiTOnlyPipeline(cfg=self.cfg, device=self.device)

    def _init_worker(self):
        super()._init_worker()
        if self.rank == 0:
            cs = int(self.cfg.chunk_size)
            self.latent_buffer = SharedTensorBuffer(
                f"{self.shm_prefix}_latent",
                frame_shape=(self.cfg.dit_config.out_channels, cs,
                             self.cfg.latent_height, self.cfg.latent_width),
                dtype=np.float32, max_len=64, create=True,
            )
            self.vae_ctrl = SharedControlBuffer(f"{self.shm_prefix}_vaectrl", create=True)
            # The VAE process sets this once its model is loaded and warmed up.
            self.vae_ready = SharedTensorBuffer(
                f"{self.shm_prefix}_vaeready", frame_shape=(1,), dtype=np.int64,
                max_len=1, create=True,
            )

    def _log_ready(self):
        # Half the pipeline lives in the VAE process, and the frontend sees no
        # frames until it is up, so the backend is not ready until it reports in.
        deadline = time.time() + VAE_READY_TIMEOUT_S
        while self.vae_ready.num == 0:
            if time.time() > deadline:
                raise RuntimeError(
                    f"the VAE process did not report ready within "
                    f"{VAE_READY_TIMEOUT_S:.0f}s; check its output for a load failure"
                )
            time.sleep(0.2)
        logger.info("serving: disaggregated DiT || VAE, dit_sp=%d (VAE group up)", self.sp_size)
        logger.info("WorldPlay backend READY")

    def _relay(self, opcode: int):
        if self.rank == 0:
            # start/reset can wait on a slow first-launch VAE process; terminate
            # (opcode 2) stays short so shutdown is prompt.
            self.vae_ctrl.send(opcode, timeout_s=(10.0 if opcode == 2 else 300.0))

    # override: ship latents over shm instead of decoding
    def _do_step(self, actions):
        if self.rank == 0:
            vm, ks, act = self._camera_decode(actions)
            self._vm_buf.copy_(vm.to(self.device, torch.float32))
            self._ks_buf.copy_(ks.to(self.device, torch.float32))
            self._act_buf.copy_(act.to(self.device, torch.float32))
        global_broadcast(self._vm_buf, src=0)
        global_broadcast(self._ks_buf, src=0)
        global_broadcast(self._act_buf, src=0)
        latents = self.pipe.step(viewmats=self._vm_buf, Ks=self._ks_buf, action=self._act_buf)
        if self.rank == 0 and latents is not None:
            self.latent_buffer.write(latents[0].to(torch.float32).cpu().numpy())

    def start(self):
        super().start()
        if self.rank == 0:
            self.latent_buffer.clear()
        self._relay(1)

    def reset(self):
        super().reset()
        if self.rank == 0:
            self.latent_buffer.clear()
        self._relay(3)

    def terminate(self):
        self._relay(2)
        if self.rank == 0:
            self.latent_buffer.unlink()
            self.vae_ctrl.unlink()
            self.vae_ready.unlink()
        super().terminate()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", required=True, help="app runtime config YAML")
    ap.add_argument("--sp", type=int, required=True, help="sequence-parallel degree for the DiT")
    ap.add_argument("--shm-prefix", required=True,
                    help="name prefix for the buffers shared with the VAE process")
    args = ap.parse_args()
    ShmDiTWorker(cfg_path=args.cfg, sp_size=args.sp, shm_prefix=args.shm_prefix).loop()


if __name__ == "__main__":
    main()
