"""Pipeline-parallel multi-GPU engine for LongLive (DiT stage ∥ VAE stage).

Splits the two IR pipeline stages across disjoint GPU sets and overlaps them
across chunks: the DiT ranks produce chunk N's clean latents and hand them to
the VAE ranks (point-to-point), then immediately start chunk N+1's DiT while the
VAE ranks decode chunk N. This is the pipeline-parallel lever the IR surfaced
(Stage 0 kv_cache ∥ Stage 1 vae_feat_cache; every denoise/cache_write op is
cross-chunk-independent of every vae_decode op).

Rank layout (world = D + V):
  * DiT ranks [0 .. D-1]: sequence-parallel DiT over an SP group of size D.
    Rank 0 is the coordinator (owns adapter/audio/ASR, drives the loop, does the
    P2P handoff to the VAE lead). DiT-internal collectives (all-to-all, noise
    broadcast, prompt-embed broadcast) run over the DiT SP group only.
  * VAE ranks [D .. D+V-1]: decode. Rank D is the VAE lead (receives latents
    from DiT rank 0, and for V>1 broadcasts them to the VAE tile group so all
    VAE ranks spatial-tile-decode via the group-scoped `vendor_vae_plan`). The
    lead owns the shm video buffer.

Custom (non-uniform) process groups are built directly on the shared
`init_model_parallel_group` primitive and registered into `parallel_state`'s
globals (`_SP`/`_TP`/`_DP`) so shared consumers (attention backends, KV cache,
runners) see a consistent world. The SP group is [DiT-ranks] + singletons; the
VAE tile group is [VAE-ranks] + singletons. `_TP`/`_DP` are trivial (per-rank)
so `model_parallel_is_initialized()` holds and `get_tp_group()` is valid.

Correctness: the DiT stage draws the identical reference noise and produces the
same clean latents; the VAE stage decodes them as the single-GPU reference
would (VAE tiling with halos is numerically exact). Warmup chunks are decoded
(to warm the VAE) but not written to the video buffer.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import torch

import wllm.serving.distributed.parallel_state as ps
from wllm.serving.distributed.parallel_state import (
    get_sp_group,
    get_world_group,
    init_distributed_environment,
    init_model_parallel_group,
)
from wllm.serving.logger import init_logger
from wllm.serving.utils.rand import set_global_seed

from wllm.apps.longlive.backend.rocm.ir.engine import ChunkIndices, LongLiveCore
from wllm.apps.longlive.backend.rocm.worker_base import LongLiveWorkerBase
from wllm.serving.channels.shm_channel.tensor_buffer import SharedTensorBuffer

logger = init_logger(__name__)

OP_STEP, OP_RESET, OP_TERM, OP_STEP_NW = 0, 1, 2, 3


def _latents_shape(cfg):
    return (1, int(cfg.dit_config.out_channels), int(cfg.chunk_size),
            int(cfg.latent_height), int(cfg.latent_width))


def _embeds_shape(cfg):
    return (1, int(cfg.max_sequence_length), int(cfg.dit_config.text_dim))


# ---------------------------------------------------------------------------
# custom non-uniform group setup
# ---------------------------------------------------------------------------
def setup_pipeline_groups(num_dit: int, num_vae: int):
    """Init the distributed world and build:
      * SP group  = [[0..D-1]] + [[D],[D+1],...]  -> DiT ranks share SP-D
      * VAE group = [[0],...,[D-1]] + [[D..D+V-1]] -> VAE ranks share the tile group
      * trivial TP/DP (per-rank) so model_parallel_is_initialized() holds
    Returns (rank, local_rank, vae_group)."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    assert world == num_dit + num_vae, (world, num_dit, num_vae)
    torch.cuda.set_device(local_rank)
    init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank,
                                 distributed_init_method="env://")
    backend = torch.distributed.get_backend(ps.get_world_group().device_group)
    lr = ps.get_world_group().local_rank

    sp_ranks = [list(range(num_dit))] + [[num_dit + i] for i in range(num_vae)]
    vae_ranks = [[i] for i in range(num_dit)] + [list(range(num_dit, world))]
    trivial = [[r] for r in range(world)]

    assert ps._SP is None
    ps._SP = init_model_parallel_group(sp_ranks, lr, backend, group_name="sp")
    ps._TP = init_model_parallel_group(trivial, lr, backend, group_name="tp")
    ps._DP = init_model_parallel_group(trivial, lr, backend, group_name="dp")
    vae_group = init_model_parallel_group(vae_ranks, lr, backend, group_name="vae_tile")
    return rank, local_rank, vae_group


def _dit_noise_hook(t: torch.Tensor) -> torch.Tensor:
    # broadcast rank-0 noise to the DiT SP group only (VAE ranks don't draw)
    get_sp_group().broadcast(t, src=0)
    return t


# ---------------------------------------------------------------------------
# DiT coordinator (world rank 0)
# ---------------------------------------------------------------------------
class PipeDiTCoordinator(LongLiveWorkerBase):
    def __init__(self, cfg_path: str, num_dit: int, num_vae: int):
        self._num_dit = num_dit
        self._num_vae = num_vae
        self._sp_size = num_dit
        self._vae_lead = num_dit
        super().__init__(cfg_path)

    def _init_gen(self):
        noise_hook = _dit_noise_hook if self._sp_size > 1 else None
        self.core = LongLiveCore(self.cfg, self.device, build_vae=False,
                                 build_text_encoder=True, build_dit=True,
                                 noise_hook=noise_hook)
        self._num_steps = int(self.cfg.num_inference_steps)
        self._chunk_size = int(self.cfg.chunk_size)

    def _create_video_buffer(self):
        return None  # the VAE lead owns the video buffer

    @torch.inference_mode()
    def _warmup(self):
        set_global_seed(self.cfg.seed)
        self.core.reset()
        self.core.seed()
        self._bcast_prompt(self.cfg.prompt or "warmup")
        self._send_cmd(OP_RESET)
        ring = max(self._chunk_size, int(self.cfg.context_window_size) + self._chunk_size)
        for _ in range((ring // self._chunk_size) + 1):
            self._pipe_step(write=False)
        self.core.reset()
        self._send_cmd(OP_RESET)
        torch.cuda.synchronize()

    def _reset_gen(self):
        self.core.reset()
        self._sp_cmd({"op": "reset"})
        self._send_cmd(OP_RESET)

    @torch.inference_mode()
    def _apply_prompt(self, text: str, is_first: bool):
        if is_first:
            self.core.reset()
            self._sp_cmd({"op": "reset"})
            self.core.seed()
            self._bcast_prompt(text)
            self._send_cmd(OP_RESET)
            logger.info("LongLivePipeline.init_session: prompt=%r block_idx=%d",
                        text, self.core.block_idx)
        else:
            self._bcast_prompt(text)
            self.core.apply_scene_cut(text)
            logger.info("LongLivePipeline.update_prompt: block_idx=%d, prompt=%r",
                        self.core.block_idx, text)

    @torch.inference_mode()
    def _step(self):
        self._pipe_step(write=True)

    def _teardown(self):
        self._send_cmd(OP_TERM)

    # -- SP coordination among DiT ranks --
    def _sp_cmd(self, cmd):
        if self._sp_size > 1:
            get_sp_group().broadcast_object(cmd, src=0)

    @torch.inference_mode()
    def _bcast_prompt(self, text: str):
        embeds = self.core._get_t5_prompt_embeds(text).contiguous()
        if self._sp_size > 1:
            self._sp_cmd({"op": "set_prompt"})
            get_sp_group().broadcast(embeds, src=0)
        self.core._prompt_embeds = embeds
        self.core.dit_runner.encode(embeds)

    # -- P2P to the VAE lead --
    def _send_cmd(self, op: int):
        payload = torch.tensor([op], dtype=torch.int64, device=self.device)
        get_world_group().send(payload, dst=self._vae_lead)

    @torch.inference_mode()
    def _pipe_step(self, write: bool = True):
        core = self.core
        idx = core.compute_chunk_indices()
        if self._sp_size > 1:
            self._sp_cmd({"op": "step", "cs": idx.cache_start, "ce": idx.cache_end,
                          "rs": idx.rope_start, "re": idx.rope_end})
        latents = core.sample_noise()
        for i in range(self._num_steps):
            latents = core.denoise_step(latents, i, idx)
        self._send_cmd(OP_STEP if write else OP_STEP_NW)
        wg = get_world_group()
        wg.send(latents.contiguous(), dst=self._vae_lead)
        # Bound the DiT->VAE pipeline depth to 1 chunk: wait for the VAE lead to
        # acknowledge it has received THIS chunk before starting the handoff of
        # the next one. Without this the DiT races many chunks ahead of the VAE
        # (async NCCL sends), so a prompt change applied at DiT-block B only
        # becomes visible ~pipeline-depth chunks later (measured ~7.8 s). The ack
        # returns as soon as the VAE has the latents (before it decodes), so the
        # VAE(N) ∥ DiT(N+1) overlap is preserved; the DiT simply cannot lap it.
        wg.recv(size=(1,), dtype=torch.int64, src=self._vae_lead)
        core.cache_write(latents, idx)
        core.advance_chunk(idx)


# ---------------------------------------------------------------------------
# DiT SP follower (world ranks 1..num_dit-1)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_dit_sp_follower(cfg, device):
    core = LongLiveCore(cfg, device, build_vae=False, build_text_encoder=False,
                        build_dit=True, noise_hook=_dit_noise_hook)
    num_steps = int(cfg.num_inference_steps)
    embeds_shape = _embeds_shape(cfg)
    sp = get_sp_group()
    while True:
        cmd = sp.broadcast_object(None, src=0)
        op = cmd["op"]
        if op == "reset":
            core.reset()
        elif op == "set_prompt":
            embeds = torch.empty(embeds_shape, device=device, dtype=core.dtype)
            sp.broadcast(embeds, src=0)
            core.dit_runner.encode(embeds)
        elif op == "step":
            idx = ChunkIndices(cmd["cs"], cmd["ce"], cmd["rs"], cmd["re"], 0, 0)
            latents = core.sample_noise()
            for i in range(num_steps):
                latents = core.denoise_step(latents, i, idx)
            core.cache_write(latents, idx)
        elif op == "term":
            break


# ---------------------------------------------------------------------------
# VAE stage (world ranks num_dit..num_dit+num_vae-1)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_vae_stage(cfg, device, num_dit: int, num_vae: int, vae_group):
    rank = ps.get_world_rank()
    is_lead = (rank == num_dit)
    tile = num_vae > 1
    core = LongLiveCore(cfg, device, build_vae=True, build_text_encoder=False,
                        build_dit=False)
    if tile:
        from wllm.apps.longlive.backend.rocm.vendor_vae_plan import patch_wan_vae_for_group
        patch_wan_vae_for_group(vae_group)
        core.vae_runner.vae.decoder.world_size = num_vae   # enter the tile path
    else:
        core.vae_runner.vae.decoder.world_size = 1
    chunk_size = int(cfg.chunk_size)
    lat_shape = _latents_shape(cfg)
    wg = get_world_group()


    video_buffer = None
    if is_lead:
        video_buffer = SharedTensorBuffer(
            name=cfg.video_buffer_name, frame_shape=(cfg.height, cfg.width, 3),
            max_len=cfg.max_num_frames, dtype=np.uint8, create=True)
    logger.info("VAE rank %d up (lead=%s, tile=%s)", rank, is_lead, tile)
    if is_lead:
        # The video buffer now exists; tell the driver it may report ready.
        wg.send(torch.tensor([1], dtype=torch.int64, device=device), dst=0)

    while True:
        # lead gets the command from DiT rank 0, then (if tiling) shares it with
        # the VAE tile group so all VAE ranks step in lockstep.
        # src=0 is the *local* index in each group -> the VAE lead is the first
        # member of vae_group; DiT rank 0 is the first member of the world group.
        if is_lead:
            op = int(wg.recv(size=(1,), dtype=torch.int64, src=0).item())
            if tile:
                vae_group.broadcast_object({"op": op}, src=0)
        else:
            op = int(vae_group.broadcast_object(None, src=0)["op"])

        if op == OP_TERM:
            break
        elif op == OP_RESET:
            core.reset()
            if video_buffer is not None:
                video_buffer.clear()
        elif op in (OP_STEP, OP_STEP_NW):
            if is_lead:
                latents = wg.recv(size=lat_shape, dtype=core.dtype, src=0)
                # ack the DiT immediately (bounds pipeline depth to 1 chunk)
                wg.send(torch.tensor([1], dtype=torch.int64, device=device), dst=0)
                if tile:
                    vae_group.broadcast(latents.contiguous(), src=0)
            else:
                latents = torch.empty(lat_shape, device=device, dtype=core.dtype)
                vae_group.broadcast(latents, src=0)
            write = (op == OP_STEP)
            for l in range(chunk_size):
                frame = core.decode_frame(latents, l)
                if write and video_buffer is not None:
                    video_buffer.write(frame.cpu().numpy())

    if video_buffer is not None:
        video_buffer.unlink()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def pipe_main(cfg_path: str, num_dit: int = 1, num_vae: int = 1, vae_tile: bool = False,
              coordinator_cls=None):
    from wllm.serving.rt_config import RTConfig
    from wllm.serving.utils.torch_utils import set_torch_options
    set_torch_options()  # TF32 + cudnn.benchmark, as the reference does

    rank, local_rank, vae_group = setup_pipeline_groups(num_dit, num_vae)
    device = torch.device("cuda", local_rank)
    cfg = RTConfig.from_yaml(cfg_path, is_path=True)

    if rank == 0:
        cls = coordinator_cls or PipeDiTCoordinator
        worker = cls(cfg_path, num_dit, num_vae)
        # The VAE group owns the video buffer on this topology, and the frontend
        # cannot attach until it exists -- so wait for the VAE lead to report in
        # before claiming the backend is up.
        ps.get_world_group().recv(size=(1,), dtype=torch.int64, src=num_dit)
        logger.info("serving: pipelined DiT || VAE (VAE group up)")
        logger.info("LongLive backend READY")
        worker.loop()
    elif rank < num_dit:
        run_dit_sp_follower(cfg, device)
    else:
        run_vae_stage(cfg, device, num_dit, num_vae, vae_group)
