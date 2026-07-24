"""Vendored from wllm/distributed/vae_plan.py (commit state 2026-06-05).

Divergence: the original split_tile/gather_tile hard-code the DEFAULT process
group (dist.get_rank()/get_world_size()/all_gather_into_tensor on the global
world). The stage-split worker tiles the VAE decoder over only the *VAE
sub-group* (a subset of ranks), so these versions take an explicit
(rank, world, group) and run the all-gather on that group. The tiling math
(HALO=13, S=8, _plan_centers for world in {2,3,4}) is unchanged, so numerical
behaviour for a given world size matches the original.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, List
import torch
import torch.distributed as dist

HALO = 13
S = 8


def _plan_centers(W: int, world_size: int) -> List[Tuple[int, int]]:
    if world_size == 2:
        mid = W // 2
        return [(0, mid), (mid, W)]
    if world_size == 3:
        cm = (W - 2 * HALO) // 3
        ce = (W - cm) // 2
        return [(0, ce), (ce, ce + cm), (ce + cm, W)]
    if world_size == 4:
        b1 = round(W / 4 + HALO / 2)
        b2 = W // 2
        b3 = W - b1
        return [(0, b1), (b1, b2), (b2, b3), (b3, W)]
    raise ValueError(f"Unsupported world_size={world_size}, expected 2, 3, or 4")


@torch.no_grad()
def split_tile(x_mid: torch.Tensor, rank: int, world: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if x_mid.dim() != 5:
        raise ValueError(f"expected x_mid [N,T,H,W,C], got {tuple(x_mid.shape)}")
    N, T, H, W, C = x_mid.shape
    centers = _plan_centers(W, world)
    w0, w1 = centers[rank]
    lw = max(w0 - HALO, 0)
    rw = min(w1 + HALO, W)
    x_piece = x_mid[:, :, :, lw:rw, :].contiguous()
    meta = {"rank": rank, "world_size": world, "w0": w0, "w1": w1, "lw": lw, "rw": rw,
            "crop_l": (w0 - lw) * S, "crop_r": (w1 - lw) * S, "mid_W": W}
    return x_piece, meta


@torch.no_grad()
def gather_tile(y_local: torch.Tensor, meta: Dict[str, Any], rank: int, world: int,
                group=None) -> torch.Tensor:
    if y_local.dim() != 5:
        raise ValueError(f"expected y_local [N,T,H,W,C], got {tuple(y_local.shape)}")
    N, T_out, H_out, Wl, C_out = y_local.shape
    crop_l = int(meta["crop_l"]); crop_r = int(meta["crop_r"])
    y_center = y_local[:, :, :, crop_l:crop_r, :].contiguous()
    Wc = y_center.shape[3]
    mid_W = int(meta["mid_W"])
    centers = _plan_centers(mid_W, world)
    out_widths = [(w1 - w0) * S for (w0, w1) in centers]
    OUT_W = sum(out_widths)
    gather_W = max(out_widths)
    y_pad = torch.zeros((N, T_out, H_out, gather_W, C_out), device=y_center.device, dtype=y_center.dtype)
    y_pad[:, :, :, :Wc, :] = y_center
    gathered = torch.empty((world, N, T_out, H_out, gather_W, C_out), device=y_pad.device, dtype=y_pad.dtype)
    dist.all_gather_into_tensor(gathered, y_pad, group=group)
    tiles = [gathered[i, :, :, :, :out_widths[i], :].contiguous() for i in range(world)]
    y_full = torch.cat(tiles, dim=3)
    return y_full
