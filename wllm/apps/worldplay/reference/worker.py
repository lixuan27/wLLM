"""Reference single-GPU sequential WorldPlay worker.

    poll WASD/arrow-key actions
        -> WorldPlay step (DiT denoise + VAE decode)
        -> write generated frames to the output video buffer

"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import wllm.kernels_t
import numpy as np
import torch

from wllm.serving.channels.shm_channel.control_buffer import SharedControlBuffer
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer
from wllm.serving.logger import init_logger
from wllm.apps.worldplay.reference.config import WorldPlayReferenceConfig
from wllm.apps.worldplay.reference.pipeline import WorldPlayPipeline
from wllm.serving.utils.rand import set_global_seed
from wllm.serving.utils.torch_utils import set_torch_options

logger = init_logger(__name__)
set_torch_options()


class WorldPlayWorker:
    """Sequential single-GPU reference worker for the WorldPlay app."""

    def __init__(self, cfg_path: str):
        self.reference_cfg = WorldPlayReferenceConfig.from_yaml(cfg_path, is_path=True)
        self.cfg = self.reference_cfg.to_runtime_config()
        self._init_worker()
        self.warmup()
        logger.info("WorldPlay backend READY")

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    def _create_pipeline(self):
        return WorldPlayPipeline(cfg=self.cfg, device=self.device)

    def _init_worker(self):
        if self.cfg.device != "cuda":
            raise ValueError(
                f"The reference worker expects cfg.device='cuda', got {self.cfg.device!r}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("The reference worker requires a visible CUDA GPU.")

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        self.pipe = self._create_pipeline()
        self.pipe.start_instance()

        # Camera-action accumulators: T tracks the running camera pose
        # (4x4 view matrix), C_inv tracks the rolling intrinsics inverse.
        self.T = torch.eye(4, dtype=torch.float32)
        self.C_inv = torch.zeros((4, 4), dtype=torch.float32)
        self.num_executed_actions = 0
        self.session_started = False

        # Control channel (start / terminate / reset opcodes).
        self.ctrl_buffer = SharedControlBuffer(
            self.cfg.ctrl_buffer_name, create=True,
        )

        # Action input buffer: discrete WASD/arrow codes (0..80).
        self.action_buffer = SharedTensorBuffer(
            self.cfg.action_buffer_name,
            frame_shape=(1,),
            dtype=np.int64,
            max_len=int(self.cfg.max_num_actions),
            create=True,
        )

    # ------------------------------------------------------------------
    # session lifecycle
    # ------------------------------------------------------------------

    def warmup(self):
        set_global_seed(self.cfg.seed)
        self.pipe.init_session(
            prompt=self.cfg.prompt,
            negative_prompt=None,
            image_path=self.cfg.image_path,
        )

        # Push enough zero-action chunks through the pipeline so cuDNN /
        # inductor caches are warm and kernels are picked, before live
        # actions arrive.
        for _ in range(int(self.cfg.max_num_actions) // int(self.cfg.chunk_size)):
            dummy_actions = np.zeros((int(self.cfg.chunk_size),), dtype=np.int64)
            self.predict(dummy_actions)

        self.T[:] = torch.eye(4, dtype=torch.float32)
        self.C_inv[:] = torch.zeros((4, 4), dtype=torch.float32)
        self.pipe.reset()

    def start(self):
        set_global_seed(self.cfg.seed)
        self.num_executed_actions = 0
        self.session_started = True

        custom_image_path = os.path.join(
            "/tmp", f"wllm_custom_img_{self.cfg.ctrl_buffer_name}.png",
        )
        image_path = (
            custom_image_path if os.path.exists(custom_image_path) else self.cfg.image_path
        )
        self.pipe.init_session(
            prompt=self.cfg.prompt,
            negative_prompt=None,
            image_path=image_path,
        )
        self.ctrl_buffer.commit()
        logger.info("WorldPlay session started")

    def reset(self):
        self.session_started = False
        self.num_executed_actions = 0
        self.T[:] = torch.eye(4, dtype=torch.float32)
        self.C_inv[:] = torch.zeros((4, 4), dtype=torch.float32)
        self.pipe.reset()
        self.action_buffer.clear()
        self.ctrl_buffer.commit()
        logger.info("WorldPlay session reset")

    def terminate(self):
        self.session_started = False
        self.pipe.terminate_instance()
        self.ctrl_buffer.unlink()
        self.action_buffer.unlink()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # one chunk: actions -> camera matrices -> DiT step -> VAE decode
    # ------------------------------------------------------------------

    def predict(self, curr_pose: List[int]) -> None:
        """Run one chunk of WorldPlay generation.

        Decodes the discrete action codes into camera (view, intrinsics,
        action-label) tensors using the shared CUDA kernels, steps the
        pipeline, and writes the chunk's decoded frames to the shared
        video buffer in a single call.
        """
        translation_codes, rotation_codes = (
            wllm.kernels_t.camera_action.decode_combined_actions(curr_pose)
        )
        curr_viewmats, curr_Ks, curr_action = (
            wllm.kernels_t.camera_action.motions_to_matrix_with_rotation(
                translation_codes,
                rotation_codes,
                self.T,
                self.C_inv,
                first_chunk=(self.pipe._session_ctx["latent_chunk_idx"] == 0),
            )
        )
        curr_action = wllm.kernels_t.camera_action.compute_worldplay_combined_label(
            curr_action, rotation_codes,
        )
        chunk_video = self.pipe.step(
            viewmats=curr_viewmats.unsqueeze(0),
            Ks=curr_Ks.unsqueeze(0),
            action=curr_action.unsqueeze(0),
        )
        if chunk_video is not None and chunk_video.shape[0] > 0:
            self.pipe._video_buffer.write(chunk_video)

    # ------------------------------------------------------------------
    # input polling
    # ------------------------------------------------------------------

    def get_actions(self) -> Optional[np.ndarray]:
        """Pull the next chunk's worth of action codes from the buffer.

        - In ``continuous_decoding`` mode we read whatever is currently
          available (1..chunk_size codes), pad / repeat to ``chunk_size``,
          and emit a zero-action chunk if nothing is ready.
        - In ``reactive_decoding`` mode we behave like continuous, but
          return ``None`` when nothing is ready (so the worker idles
          instead of generating dead frames).
        - Otherwise we wait for a full ``chunk_size`` worth of actions
          before stepping.
        """
        if self.cfg.continuous_decoding or self.cfg.reactive_decoding:
            actions: list = []
            for _ in range(int(self.cfg.chunk_size)):
                self.num_executed_actions, new_action = self.action_buffer.read(
                    self.num_executed_actions, 1,
                )
                if new_action is None:
                    break
                actions.append(new_action.ravel())

            if len(actions) == 0:
                if self.cfg.continuous_decoding:
                    actions = np.zeros((int(self.cfg.chunk_size),), dtype=np.int64)
                else:
                    actions = None
            else:
                actions = np.concatenate(actions).ravel()
                base, rem = divmod(int(self.cfg.chunk_size), len(actions))
                repeats = np.full(len(actions), base, dtype=np.int64)
                repeats[:rem] += 1
                actions = np.repeat(actions, repeats)
        else:
            self.num_executed_actions, actions = self.action_buffer.read(
                self.num_executed_actions, int(self.cfg.chunk_size),
            )

        if actions is not None:
            actions = actions.flatten()
            if self.num_executed_actions > int(self.cfg.max_num_actions):
                return None
        return actions

    # ------------------------------------------------------------------
    # control buffer + main loop
    # ------------------------------------------------------------------

    def is_start(self):
        return int(self.ctrl_buffer.recv()) == 1

    def is_terminate(self):
        return int(self.ctrl_buffer.recv()) == 2

    def is_reset(self):
        return int(self.ctrl_buffer.recv()) == 3

    def loop(self):
        while True:
            if self.is_terminate() and self.session_started:
                self.terminate()
                break
            elif self.is_start() and not self.session_started:
                self.start()
            elif self.is_reset() and self.session_started:
                self.reset()

            if self.session_started:
                curr_actions = self.get_actions()
                if curr_actions is not None:
                    self.predict(curr_actions)
            else:
                time.sleep(0.005)
