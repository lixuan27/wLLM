"""Single source of truth for LongLive per-chunk generation math.

These functions mirror, op-for-op, the reference ``LongLivePipeline.step()``
in ``wllm/apps/longlive/reference/pipeline.py``. Both the IR operators
(``wllm/apps/longlive/backend/ir/ops.py``) and the deployment-variant pipelines
(``wllm/apps/longlive/backend/pipeline.py``) call these so that the
fine-grained IR decomposition and the multi-GPU variants cannot drift from
the reference numerics.

A "core" passed to these functions is any object exposing:
    cfg, device, dtype, dit_runner, vae_runner, timesteps, sigmas,
    noise_generator
A "ring" is a plain dict carrying the LongLive sliding-window bookkeeping
(block_idx, rolling_writes, max_filled_slot, pinned_slot, shot_index,
temporal_offset_latents, scene_cut_pending, latent_decode_count).
"""
from __future__ import annotations

from typing import Any, Dict

import torch


# ---------------------------------------------------------------------------
# ring-state bookkeeping
# ---------------------------------------------------------------------------

def new_ring_state() -> Dict[str, int]:
    return dict(
        block_idx=0,
        rolling_writes=0,
        max_filled_slot=-1,
        pinned_slot=-1,
        shot_index=0,
        temporal_offset_latents=0,
        scene_cut_pending=False,
        latent_decode_count=0,
    )


def ring_capacity_latents(cfg) -> int:
    chunk = int(cfg.chunk_size)
    return max(chunk, int(cfg.context_window_size) + chunk)


def sink_chunks(cfg) -> int:
    return int(cfg.sink_size) // int(cfg.chunk_size)


def plan_chunk(core, ring: Dict[str, Any]) -> Dict[str, int]:
    """Compute KV-cache write slot + rope window for the current chunk.

    Mirrors the slot-selection / rope arithmetic at the top of
    ``LongLivePipeline.step()`` exactly.
    """
    cfg = core.cfg
    chunk_size = int(cfg.chunk_size)
    kv_spatial = int(cfg.kv_spatial)
    chunk_tokens = chunk_size * kv_spatial
    rc_latents = ring_capacity_latents(cfg)
    rc_chunks = rc_latents // chunk_size
    rc_tokens = rc_chunks * chunk_tokens
    s_chunks = sink_chunks(cfg)
    rolling_capacity_chunks = rc_chunks - s_chunks
    assert rolling_capacity_chunks > 0, (
        f"sink leaves no rolling room: sink_chunks={s_chunks} >= rc_chunks={rc_chunks}"
    )

    block_idx = ring["block_idx"]
    pinned_slot = ring["pinned_slot"]
    rolling_writes = ring["rolling_writes"]

    if block_idx < s_chunks:
        cache_chunk_idx = block_idx
    elif pinned_slot < 0:
        cache_chunk_idx = s_chunks + (rolling_writes % rolling_capacity_chunks)
    else:
        available_slots = [
            s_chunks + i for i in range(rolling_capacity_chunks)
            if (s_chunks + i) != pinned_slot
        ]
        assert available_slots, "no rolling slots left after pinning"
        cache_chunk_idx = available_slots[rolling_writes % len(available_slots)]

    cache_start = cache_chunk_idx * chunk_tokens
    new_max_filled_slot = max(ring["max_filled_slot"], cache_chunk_idx)
    cache_end = (new_max_filled_slot + 1) * chunk_tokens

    global_start_latent = block_idx * chunk_size + ring["temporal_offset_latents"]
    rope_start = global_start_latent * kv_spatial
    rope_end = rope_start + chunk_tokens

    assert cache_start + chunk_tokens <= rc_tokens
    assert cache_end <= rc_tokens
    return dict(
        cache_chunk_idx=cache_chunk_idx,
        cache_start=cache_start,
        cache_end=cache_end,
        rope_start=rope_start,
        rope_end=rope_end,
        new_max_filled_slot=new_max_filled_slot,
    )


def advance_ring(core, ring: Dict[str, Any], plan: Dict[str, int]) -> None:
    """Advance the ring bookkeeping after a chunk is committed (mirrors the
    tail of ``LongLivePipeline.step()``)."""
    cfg = core.cfg
    ring["max_filled_slot"] = plan["new_max_filled_slot"]
    if ring["block_idx"] >= sink_chunks(cfg):
        ring["rolling_writes"] += 1
    if ring["scene_cut_pending"]:
        ring["pinned_slot"] = plan["cache_chunk_idx"]
        ring["scene_cut_pending"] = False
    ring["block_idx"] += 1


# ---------------------------------------------------------------------------
# compute primitives
# ---------------------------------------------------------------------------

def sample_noise(core, shape) -> torch.Tensor:
    return torch.randn(shape, device=core.device, dtype=core.dtype,
                       generator=core.noise_generator)


def initial_noise(core) -> torch.Tensor:
    cfg = core.cfg
    noise_shape = (1, cfg.dit_config.out_channels, int(cfg.chunk_size),
                   cfg.latent_height, cfg.latent_width)
    return sample_noise(core, noise_shape)


def denoise_one_step(core, latents: torch.Tensor, plan: Dict[str, int],
                     step_idx: int) -> torch.Tensor:
    """One DMD denoise step. Mirrors the body of the ``for i in range(num_steps)``
    loop in the reference (is_cache=False forward + fp64 x0 conversion + the
    fresh-noise re-noising for all but the last step)."""
    cfg = core.cfg
    chunk_size = int(cfg.chunk_size)
    t = core.timesteps[step_idx]
    sigma_t = core.sigmas[step_idx]
    timestep = torch.full((chunk_size,), t.item(), device=core.device,
                          dtype=torch.float32)
    flow_pred = core.dit_runner.run(
        latents=latents, timestep=timestep, is_cache=False,
        cache_start=plan["cache_start"], cache_end=plan["cache_end"],
        rope_start=plan["rope_start"], rope_end=plan["rope_end"],
    )
    x0_pred = (
        latents.to(torch.float64) - sigma_t.to(torch.float64) * flow_pred.to(torch.float64)
    ).to(latents.dtype)
    num_steps = len(core.timesteps)
    if step_idx < num_steps - 1:
        sigma_next = core.sigmas[step_idx + 1].to(torch.float64)
        fresh_noise = sample_noise(core, latents.shape)
        latents = (
            (1.0 - sigma_next) * x0_pred.to(torch.float64)
            + sigma_next * fresh_noise.to(torch.float64)
        ).to(latents.dtype)
    else:
        latents = x0_pred
    return latents


def write_clean_cache(core, latents: torch.Tensor, plan: Dict[str, int]) -> None:
    """t=0 pass: write the chunk's clean K/V into the ring (is_cache=True)."""
    chunk_size = int(core.cfg.chunk_size)
    zero_timestep = torch.zeros((chunk_size,), device=core.device, dtype=torch.float32)
    core.dit_runner.run(
        latents=latents, timestep=zero_timestep, is_cache=True,
        cache_start=plan["cache_start"], cache_end=plan["cache_end"],
        rope_start=plan["rope_start"], rope_end=plan["rope_end"],
    )


def decode_latent_frame(core, latents: torch.Tensor, l: int, is_first: bool):
    """Decode a single latent frame to a (T,H,W,3) uint8 numpy stack."""
    latent_one = latents[:, :, l:l + 1].contiguous()
    video_chunk = core.vae_runner.run(latent_one, is_first)
    return video_chunk[0]
