# tile_parallel_wan_decoder_single_collective.py
# One-collective (single all_gather) final stitch.
#
# Layout:
#   x_mid   : [N, 1, 44, 80, 1024] (N,T,H,W,C)
#   y_local : [N, T_out, 352, W_local, 12]
#   y_full  : [N, T_out, 352, 640, 12]
#
# Parallel:
#   split once at x_mid along W with HALO=13, remain_scale=8
#   decode locally
#   gather_tile: ONE collective: dist.all_gather on padded y_center (no meta gather, no shape gather)

from __future__ import annotations
from typing import Dict, Any, Tuple, List

import torch
import torch.distributed as dist

# ---------- hardcoded constants ----------
HALO = 13
S = 8


def _plan_centers(W: int, world_size: int) -> List[Tuple[int, int]]:
    """
    Compute uniform center splits on a given W dimension.

    ws=2: uniform centers
    ws=3: balanced-piece centers — middle rank has two halos (one on each
        side) while the end ranks each have only one (the outer halo gets
        clipped at the image boundary), so the middle's center is HALO
        smaller than each end's. Yields piece widths e.g. 39/26/39 for W=104.
    ws=4: balanced-piece centers to equalize compute (piece widths ~39/40/40/39 with halo=13)
    """
    if world_size == 2:
        mid = W // 2
        return [(0, mid), (mid, W)]
    if world_size == 3:
        cm = (W - 2 * HALO) // 3        # center for the middle rank
        ce = (W - cm) // 2              # center for each end rank (absorbs ±1 remainder)
        return [(0, ce), (ce, ce + cm), (ce + cm, W)]
    if world_size == 4:
        b1 = round(W / 4 + HALO / 2)
        b2 = W // 2
        b3 = W - b1
        return [(0, b1), (b1, b2), (b2, b3), (b3, W)]
    raise ValueError(f"Unsupported world_size={world_size}, expected 2, 3, or 4")


@torch.no_grad()
def split_tile(x_mid: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Split x_mid along W with halo. Supports world_size in {2, 3, 4}; the
    set of valid sizes is determined by ``_plan_centers``.
    Input:
      x_mid: [N,1,44,80,1024]
    Return:
      x_piece: [N,1,44,W_piece,1024]
      meta: contains crop indices for y_local -> y_center
    """
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("split_tile requires torch.distributed to be initialized")

    rank = dist.get_rank()
    world = dist.get_world_size()

    if x_mid.dim() != 5:
        raise ValueError(f"expected x_mid [N,T,H,W,C], got {tuple(x_mid.shape)}")
    N, T, H, W, C = x_mid.shape

    centers = _plan_centers(W, world)  # raises on unsupported world_size
    w0, w1 = centers[rank]

    lw = max(w0 - HALO, 0)
    rw = min(w1 + HALO, W)

    x_piece = x_mid[:, :, :, lw:rw, :].contiguous()

    meta: Dict[str, Any] = {
        "rank": rank,
        "world_size": world,
        "w0": w0, "w1": w1,
        "lw": lw, "rw": rw,
        # crop inside y_local (output-W units)
        "crop_l": (w0 - lw) * S,
        "crop_r": (w1 - lw) * S,
        # retain W to recompute centers
        "mid_W": W,
    }
    return x_piece, meta


@torch.no_grad()
def gather_tile(y_local: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
    """
    ONE-collective gather+stitch using dist.all_gather_into_tensor.

    Input:
      y_local: [N, T_out, 352, W_local, 12]
      meta: from split_tile (crop_l/crop_r)

    Output:
      y_full: [N, T_out, 352, 640, 12]
    """
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("gather_tile requires torch.distributed to be initialized")

    rank = dist.get_rank()
    world = dist.get_world_size()

    if y_local.dim() != 5:
        raise ValueError(f"expected y_local [N,T,H,W,C], got {tuple(y_local.shape)}")
    N, T_out, H_out, Wl, C_out = y_local.shape

    # local crop to center
    crop_l = int(meta["crop_l"])
    crop_r = int(meta["crop_r"])
    if not (0 <= crop_l <= crop_r <= Wl):
        raise ValueError(f"rank{rank}: crop out of bounds crop=({crop_l},{crop_r}), Wl={Wl}")
    y_center = y_local[:, :, :, crop_l:crop_r, :].contiguous()  # [N,T,352,Wc,12]
    Wc = y_center.shape[3]

    # recompute centers from stored mid_W
    mid_W = int(meta["mid_W"])
    centers = _plan_centers(mid_W, world)
    out_widths = [(w1 - w0) * S for (w0, w1) in centers]
    OUT_W = sum(out_widths)
    gather_W = max(out_widths)
    expected_Wc = out_widths[rank]
    if Wc != expected_Wc:
        raise ValueError(f"rank{rank}: y_center width {Wc} != expected {expected_Wc}")

    # pad to uniform gather shape
    y_pad = torch.zeros((N, T_out, H_out, gather_W, C_out), device=y_center.device, dtype=y_center.dtype)
    y_pad[:, :, :, :Wc, :] = y_center

    # allocate output tensor for all ranks
    gathered = torch.empty((world, N, T_out, H_out, gather_W, C_out), device=y_pad.device, dtype=y_pad.dtype)

    # SINGLE collective into tensor
    dist.all_gather_into_tensor(gathered, y_pad)

    # unpad each rank tile and stitch along W
    tiles = [gathered[i, :, :, :, :out_widths[i], :].contiguous() for i in range(world)]
    y_full = torch.cat(tiles, dim=3)  # concat W

    expected_shape = (N, T_out, H_out, OUT_W, C_out)
    if y_full.shape != expected_shape:
        raise ValueError(f"stitched shape {tuple(y_full.shape)} != {expected_shape}")
    return y_full
