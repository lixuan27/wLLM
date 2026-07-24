"""Co-located sequence-parallel WorldPlay worker.

Launched with ``torch.distributed.run --nproc_per_node=N`` by
``wllm/apps/worldplay/backend/launch.py``. All N ranks form one SP world:
the DiT runs Ulysses sequence parallel and the VAE decoder runs ``vae_plan``
spatial tile-parallel, both over the WORLD group. Rank 0 owns the
shared-memory IPC (ctrl/action/video), reads actions, decodes the camera
matrices, and broadcasts them to every rank; all ranks step the pipeline
collectively; rank 0 writes the frames out.

Lock-step control protocol: every loop iteration rank 0 broadcasts a 2-int
packet `[op, has_step]` (op: 0 none / 1 start / 2 terminate / 3 reset); on a
step it then broadcasts the (viewmats, Ks, action) tensors so every rank feeds
the identical conditioning into the SP shard.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import numpy as np
import torch

import wllm.kernels_t

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    get_world_rank, get_world_size, get_local_torch_device,
)
from wllm.serving.distributed.communication_op import (
    global_broadcast, global_barrier, warmup_sequence_parallel_communication,
)
from wllm.serving.logger import init_logger
from wllm.apps.worldplay.reference.config import WorldPlayReferenceConfig
from wllm.serving.utils.rand import set_global_seed
from wllm.serving.utils.torch_utils import set_torch_options

from wllm.apps.worldplay.backend.rocm.runtime.colocated_pipeline import (
    ColocatedPipeline, StreamingPipeline,
)

logger = init_logger(__name__)
set_torch_options()


class ColocatedSPWorker:
    def __init__(self, cfg_path: str, sp_size: int, vae_tile: bool = True,
                 stream: bool = False):
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=sp_size)
        self.rank = get_world_rank()
        self.world = get_world_size()
        self.sp_size = sp_size
        self.vae_tile = vae_tile
        self.stream = stream
        self.device = get_local_torch_device()
        torch.cuda.set_device(self.device)

        self.reference_cfg = WorldPlayReferenceConfig.from_yaml(cfg_path, is_path=True)
        self.cfg = self.reference_cfg.to_runtime_config()

        self._init_worker()
        self.warmup()
        global_barrier()
        if self.rank == 0:
            self._log_ready()

    def _log_ready(self):
        logger.info(
            "serving: co-located worker world=%d sp=%d vae_tile=%s stream_vae=%s",
            self.world, self.sp_size, self.vae_tile, self.stream,
        )
        logger.info("WorldPlay backend READY")

    # ------------------------------------------------------------------
    def _make_pipeline(self):
        cls = StreamingPipeline if self.stream else ColocatedPipeline
        return cls(cfg=self.cfg, device=self.device)

    def _init_worker(self):
        self.pipe = self._make_pipeline()
        if not self.vae_tile:
            # isolate DiT-SP: force the VAE decoder to run full-frame (no
            # vae_plan spatial tiling) by telling it world_size==1. Set before
            # start_instance so the VAE warmup also runs untiled.
            self.pipe.vae_runner.vae.decoder.world_size = 1
        self.pipe.start_instance()             # collective: VAE warmup + latent broadcast + rank0 video buffer
        warmup_sequence_parallel_communication(self.device)

        self.T = torch.eye(4, dtype=torch.float32)
        self.C_inv = torch.zeros((4, 4), dtype=torch.float32)
        self.num_executed_actions = 0
        self.session_started = False

        cs = int(self.cfg.chunk_size)
        self._vm_buf = torch.zeros(1, cs, 4, 4, device=self.device, dtype=torch.float32)
        self._ks_buf = torch.zeros(1, cs, 3, 3, device=self.device, dtype=torch.float32)
        self._act_buf = torch.zeros(1, cs, device=self.device, dtype=torch.float32)
        self._ctrl_pkt = torch.zeros(2, dtype=torch.int64, device=self.device)

        if self.rank == 0:
            self.ctrl_buffer = SharedControlBuffer(self.cfg.ctrl_buffer_name, create=True)
            self.action_buffer = SharedTensorBuffer(
                self.cfg.action_buffer_name, frame_shape=(1,), dtype=np.int64,
                max_len=int(self.cfg.max_num_actions), create=True,
            )

    # ------------------------------------------------------------------
    # camera decode + one collective step
    # ------------------------------------------------------------------
    def _camera_decode(self, actions: np.ndarray):
        translation_codes, rotation_codes = (
            wllm.kernels_t.camera_action.decode_combined_actions(actions)
        )
        curr_viewmats, curr_Ks, curr_action = (
            wllm.kernels_t.camera_action.motions_to_matrix_with_rotation(
                translation_codes, rotation_codes, self.T, self.C_inv,
                first_chunk=(self.pipe._session_ctx["latent_chunk_idx"] == 0),
            )
        )
        curr_action = wllm.kernels_t.camera_action.compute_worldplay_combined_label(
            curr_action, rotation_codes,
        )
        return curr_viewmats.unsqueeze(0), curr_Ks.unsqueeze(0), curr_action.unsqueeze(0)

    def _do_step(self, actions: Optional[np.ndarray]):
        if self.rank == 0:
            vm, ks, act = self._camera_decode(actions)
            self._vm_buf.copy_(vm.to(self.device, torch.float32))
            self._ks_buf.copy_(ks.to(self.device, torch.float32))
            self._act_buf.copy_(act.to(self.device, torch.float32))
        global_broadcast(self._vm_buf, src=0)
        global_broadcast(self._ks_buf, src=0)
        global_broadcast(self._act_buf, src=0)
        video = self.pipe.step(viewmats=self._vm_buf, Ks=self._ks_buf, action=self._act_buf)
        if self.rank == 0 and video is not None and video.shape[0] > 0:
            self.pipe._video_buffer.write(video)

    # ------------------------------------------------------------------
    # session lifecycle (all collective)
    # ------------------------------------------------------------------
    def warmup(self):
        set_global_seed(self.cfg.seed)
        self.pipe.init_session(prompt=self.cfg.prompt, negative_prompt=None,
                               image_path=self.cfg.image_path)
        for _ in range(int(self.cfg.max_num_actions) // int(self.cfg.chunk_size)):
            self._do_step(np.zeros((int(self.cfg.chunk_size),), dtype=np.int64))
        self.T[:] = torch.eye(4, dtype=torch.float32)
        self.C_inv[:] = torch.zeros((4, 4), dtype=torch.float32)
        self.pipe.reset()

    def start(self):
        set_global_seed(self.cfg.seed)
        self.num_executed_actions = 0
        self.session_started = True
        custom = os.path.join("/tmp", f"wllm_custom_img_{self.cfg.ctrl_buffer_name}.png")
        image_path = custom if os.path.exists(custom) else self.cfg.image_path
        self.pipe.init_session(prompt=self.cfg.prompt, negative_prompt=None, image_path=image_path)
        if self.rank == 0:
            self.ctrl_buffer.commit()
            logger.info("WorldPlay session started")

    def reset(self):
        self.session_started = False
        self.num_executed_actions = 0
        self.T[:] = torch.eye(4, dtype=torch.float32)
        self.C_inv[:] = torch.zeros((4, 4), dtype=torch.float32)
        self.pipe.reset()
        if self.rank == 0:
            self.action_buffer.clear()
            self.ctrl_buffer.commit()

    def terminate(self):
        self.session_started = False
        if self.rank == 0:
            self.ctrl_buffer.commit()   # ack terminate so the adapter returns promptly
        self.pipe.terminate_instance()
        if self.rank == 0:
            self.ctrl_buffer.unlink()
            self.action_buffer.unlink()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # input polling (rank 0 only)
    # ------------------------------------------------------------------
    def get_actions(self) -> Optional[np.ndarray]:
        actions: list = []
        for _ in range(int(self.cfg.chunk_size)):
            self.num_executed_actions, new_action = self.action_buffer.read(self.num_executed_actions, 1)
            if new_action is None:
                break
            actions.append(new_action.ravel())
        if len(actions) == 0:
            return None
        actions = np.concatenate(actions).ravel()
        base, rem = divmod(int(self.cfg.chunk_size), len(actions))
        repeats = np.full(len(actions), base, dtype=np.int64)
        repeats[:rem] += 1
        actions = np.repeat(actions, repeats).flatten()
        if self.num_executed_actions > int(self.cfg.max_num_actions):
            return None
        return actions

    def is_start(self):
        return int(self.ctrl_buffer.recv()) == 1

    def is_terminate(self):
        return int(self.ctrl_buffer.recv()) == 2

    def is_reset(self):
        return int(self.ctrl_buffer.recv()) == 3

    # ------------------------------------------------------------------
    def loop(self):
        while True:
            op = 0
            has_step = 0
            actions = None
            if self.rank == 0:
                if self.is_terminate() and self.session_started:
                    op = 2
                elif self.is_start() and not self.session_started:
                    op = 1
                elif self.is_reset() and self.session_started:
                    op = 3
                if op == 0 and self.session_started:
                    actions = self.get_actions()
                    if actions is not None:
                        has_step = 1
                self._ctrl_pkt[0] = op
                self._ctrl_pkt[1] = has_step
            global_broadcast(self._ctrl_pkt, src=0)
            op = int(self._ctrl_pkt[0].item())
            has_step = int(self._ctrl_pkt[1].item())

            if op == 2:
                self.terminate()
                global_barrier()
                break
            elif op == 1:
                self.start()
            elif op == 3:
                self.reset()

            if has_step:
                self._do_step(actions)
            elif self.rank == 0:
                time.sleep(0.005 if not self.session_started else 0.0005)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", required=True, help="app runtime config YAML")
    ap.add_argument("--sp", type=int, required=True, help="sequence-parallel degree for the DiT")
    ap.add_argument("--stream-vae", action="store_true",
                    help="write each latent's frames as it is decoded instead of once per chunk")
    ap.add_argument("--no-vae-tile", action="store_true",
                    help="decode the VAE full-frame instead of tile-parallel")
    args = ap.parse_args()
    ColocatedSPWorker(
        cfg_path=args.cfg, sp_size=args.sp,
        vae_tile=not args.no_vae_tile, stream=args.stream_vae,
    ).loop()


if __name__ == "__main__":
    main()
