"""Sequence-parallel / tile-parallel multi-GPU engine for LongLive.

One `torchrun` job of N ranks in a single `torch.distributed` world, built on
the shared runtime's `parallel_state` (init_distributed_environment +
initialize_model_parallel) — no hand-rolled process groups. Rank 0 is the
**coordinator**: it owns the shared adapter (audio/ASR/video buffers) and the
async loop, and broadcasts commands so the follower ranks run the same collective
DiT/VAE forwards in lockstep. This is the shared substrate for:

  * `dit_spN`  — DiT sequence-parallel over N GPUs (SP group = all N), VAE on
    rank 0 only (tiling disabled). Isolates the DiT-SP latency lever.
  * `unified_spN` — DiT-SP over N **and** VAE spatial-tile over N (world group),
    every rank on the same chunk. Combines both model-parallel levers.
  * `vae_tileN` — DiT replicated (SP size 1) + VAE tile over N. Isolates the
    VAE-tile lever.

Which mode a rank runs is set by (sp_size, vae_mode):
  vae_mode="rank0"  -> VAE decode on rank 0 only, `decoder.world_size=1`.
  vae_mode="tile"   -> VAE decode on all ranks (Wan `split_tile`/`gather_tile`).

Correctness: rank 0 draws the *identical* noise the reference draws (same seed,
same order) and broadcasts it to all ranks over the world group (the core's
`noise_hook`), so every rank shards the same full noise and rank 0's decoded
output matches the reference. Chunk ring/RoPE indices are computed on rank 0 and
sent with each STEP command, so followers stay stateless w.r.t. session counters.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from wllm.serving.distributed.parallel_state import (
    get_world_group,
    get_world_rank,
    get_world_size,
    init_distributed_environment,
    initialize_model_parallel,
)
from wllm.serving.logger import init_logger
from wllm.serving.utils.rand import set_global_seed

from wllm.apps.longlive.backend.rocm.ir.engine import ChunkIndices, LongLiveCore
from wllm.apps.longlive.backend.rocm.worker_base import LongLiveWorkerBase

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# distributed setup
# ---------------------------------------------------------------------------
def setup_sp(sp_size: int):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        world_size=world_size, rank=rank, local_rank=local_rank,
        distributed_init_method="env://",
    )
    initialize_model_parallel(sequence_model_parallel_size=sp_size)
    return rank, world_size, local_rank


def _noise_hook(t: torch.Tensor) -> torch.Tensor:
    # broadcast rank-0's noise to every rank over the world group (spans all
    # ranks regardless of the SP-group size)
    get_world_group().broadcast(t, src=0)
    return t


def build_core(cfg, device, vae_mode: str, is_rank0: bool) -> LongLiveCore:
    build_vae = is_rank0 or (vae_mode == "tile")
    core = LongLiveCore(
        cfg, device,
        build_text_encoder=is_rank0,   # only rank 0 tokenizes+encodes prompts
        build_vae=build_vae,
        build_dit=True,
        noise_hook=_noise_hook,
    )
    if vae_mode == "rank0" and core.vae_runner is not None:
        # decode on rank 0 alone -> force the single-GPU decoder path
        core.vae_runner.vae.decoder.world_size = 1
    return core


def _embeds_shape(cfg):
    return (1, int(cfg.max_sequence_length), int(cfg.dit_config.text_dim))


# ---------------------------------------------------------------------------
# coordinator (rank 0)
# ---------------------------------------------------------------------------
class SPCoordinator(LongLiveWorkerBase):
    def __init__(self, cfg_path: str, sp_size: int, vae_mode: str):
        self._sp_size = sp_size
        self._vae_mode = vae_mode
        super().__init__(cfg_path)

    # -- generation hooks --
    def _init_gen(self):
        self.core = build_core(self.cfg, self.device, self._vae_mode, is_rank0=True)
        self.video_buffer = self._create_video_buffer()
        self._num_steps = int(self.cfg.num_inference_steps)
        self._chunk_size = int(self.cfg.chunk_size)

    @torch.inference_mode()
    def _warmup(self):
        set_global_seed(self.cfg.seed)
        self.core.reset()
        self.core.seed()
        self._bcast_set_prompt(self.cfg.prompt or "warmup")
        ring = max(self._chunk_size, int(self.cfg.context_window_size) + self._chunk_size)
        for _ in range((ring // self._chunk_size) + 1):
            self._bcast_step(write=False)
        self._bcast_reset()
        self.video_buffer.clear()

    def _reset_gen(self):
        self._bcast_reset()
        self.video_buffer.clear()

    @torch.inference_mode()
    def _apply_prompt(self, text: str, is_first: bool):
        if is_first:
            self._bcast_reset()
            self.core.seed()
            self._bcast_set_prompt(text)
            logger.info("LongLivePipeline.init_session: prompt=%r block_idx=%d",
                        text, self.core.block_idx)
        else:
            self._bcast_set_prompt(text)
            self.core.apply_scene_cut(text)
            logger.info("LongLivePipeline.update_prompt: block_idx=%d, prompt=%r",
                        self.core.block_idx, text)

    @torch.inference_mode()
    def _step(self):
        self._bcast_step(write=True)

    def _teardown(self):
        self._bcast_cmd({"op": "terminate"})
        if getattr(self, "video_buffer", None) is not None:
            self.video_buffer.unlink()

    # -- collective coordination --
    def _bcast_cmd(self, cmd: dict):
        get_world_group().broadcast_object(cmd, src=0)

    def _bcast_reset(self):
        self._bcast_cmd({"op": "reset"})
        self.core.reset()

    @torch.inference_mode()
    def _bcast_set_prompt(self, text: str):
        embeds = self.core._get_t5_prompt_embeds(text).contiguous()
        self._bcast_cmd({"op": "set_prompt"})
        get_world_group().broadcast(embeds, src=0)
        self.core._prompt_embeds = embeds
        self.core.dit_runner.encode(embeds)

    @torch.inference_mode()
    def _bcast_step(self, write: bool):
        idx = self.core.compute_chunk_indices()
        self._bcast_cmd({
            "op": "step", "cs": idx.cache_start, "ce": idx.cache_end,
            "rs": idx.rope_start, "re": idx.rope_end,
        })
        self._run_step(idx, write)
        self.core.advance_chunk(idx)

    def _run_step(self, idx: ChunkIndices, write: bool):
        core = self.core
        latents = core.sample_noise()
        for i in range(self._num_steps):
            latents = core.denoise_step(latents, i, idx)
        core.cache_write(latents, idx)
        for l in range(self._chunk_size):
            frame = core.decode_frame(latents, l)
            if write:
                self.video_buffer.write(frame.cpu().numpy())


# ---------------------------------------------------------------------------
# follower (rank > 0)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_follower(cfg, device, sp_size: int, vae_mode: str):
    core = build_core(cfg, device, vae_mode, is_rank0=False)
    num_steps = int(cfg.num_inference_steps)
    chunk_size = int(cfg.chunk_size)
    embeds_shape = _embeds_shape(cfg)
    decodes = (vae_mode == "tile")

    world = get_world_group()
    while True:
        cmd = world.broadcast_object(None, src=0)
        op = cmd["op"]
        if op == "terminate":
            break
        elif op == "reset":
            core.reset()
        elif op == "set_prompt":
            embeds = torch.empty(embeds_shape, device=device, dtype=core.dtype)
            world.broadcast(embeds, src=0)
            core.dit_runner.encode(embeds)
        elif op == "step":
            idx = ChunkIndices(cmd["cs"], cmd["ce"], cmd["rs"], cmd["re"], 0, 0)
            latents = core.sample_noise()
            for i in range(num_steps):
                latents = core.denoise_step(latents, i, idx)
            core.cache_write(latents, idx)
            if decodes:
                for l in range(chunk_size):
                    core.decode_frame(latents, l)


# ---------------------------------------------------------------------------
# entry point (called by run_worker_dist.py on every rank)
# ---------------------------------------------------------------------------
def sp_main(cfg_path: str, sp_size: int, vae_mode: str):
    from wllm.serving.rt_config import RTConfig
    from wllm.serving.utils.torch_utils import set_torch_options
    set_torch_options()  # TF32 + cudnn.benchmark, as the reference does

    rank, world_size, local_rank = setup_sp(sp_size)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        worker = SPCoordinator(cfg_path, sp_size, vae_mode)
        # rank 0 owns the video buffer here, so once it is constructed the whole
        # backend is up.
        logger.info("LongLive backend READY")
        worker.loop()
    else:
        cfg = RTConfig.from_yaml(cfg_path, is_path=True)
        run_follower(cfg, device, sp_size, vae_mode)
