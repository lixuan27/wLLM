import torch
import triton
from wllm.kernels_t.autotune import configs_for_platform
import triton.language as tl


def _dupup_cfgs():
    cfgs = []
    for BC in (4, 8):
        for BH in (1,):
            for BW in (128, 256):
                for RB in (2, 4, 8):
                    for warps in (4, 8):
                        cfgs.append(
                            triton.Config(
                                {"BLOCK_C": BC, "BLOCK_H": BH, "BLOCK_W": BW, "R_BLOCK": RB},
                                num_warps=warps,
                                num_stages=2,
                            )
                        )
    return cfgs


@triton.autotune(
    configs=configs_for_platform(_dupup_cfgs()),
    key=["C_in", "T_in", "H_in", "W_in", "t_offset", "repeats"],
)
@triton.jit
def _dupup_fused_add_vecR_kernel(
    x_ptr, h_ptr, out_ptr,
    B: tl.constexpr,
    C_in: tl.constexpr,
    T_in: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    C_out: tl.constexpr,
    factor_t: tl.constexpr,
    factor_s: tl.constexpr,
    repeats: tl.constexpr,
    t_offset: tl.constexpr,
    T_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    sxB, sxC, sxT, sxH, sxW,
    shB, shC, shT, shH, shW,
    soB, soC, soT, soH, soW,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    R_BLOCK: tl.constexpr,
):
    # 3D grid: (pid0, pid1, pid2) = (B*T*C_tiles*H_tiles, W_tiles, R_tiles)
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)

    C_tiles = tl.cdiv(C_in, BLOCK_C)
    H_tiles = tl.cdiv(H_in, BLOCK_H)

    # pid0 -> (b, t_in, c_tile, h_tile)
    tmp = pid0
    h_tile = tmp % H_tiles
    tmp //= H_tiles
    c_tile = tmp % C_tiles
    tmp //= C_tiles
    t_in = tmp % T_in
    b = tmp // T_in

    c0 = c_tile * BLOCK_C
    h0 = h_tile * BLOCK_H
    w0 = pid1 * BLOCK_W
    r0 = pid2 * R_BLOCK

    # indices
    c_idx = c0 + tl.arange(0, BLOCK_C)[:, None, None, None]   # [BC,1,1,1]
    h_idx = h0 + tl.arange(0, BLOCK_H)[None, :, None, None]   # [1,BH,1,1]
    w_idx = w0 + tl.arange(0, BLOCK_W)[None, None, :, None]   # [1,1,BW,1]
    r_idx = r0 + tl.arange(0, R_BLOCK)[None, None, None, :]   # [1,1,1,RB]

    # masks
    c_ok = c_idx < C_in
    h_ok = h_idx < H_in
    w_ok = w_idx < W_in
    r_ok = r_idx < repeats
    x_mask = c_ok & h_ok & w_ok  # [BC,BH,BW,1]

    # load x once: [BC,BH,BW,1]
    x_off = (
        b * sxB
        + c_idx * sxC
        + t_in * sxT
        + h_idx * sxH
        + w_idx * sxW
    )
    x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0)

    # vectorized repeats mapping (no for-loop)
    factor = factor_t * factor_s * factor_s

    # c_rep = c*repeats + r
    c_rep = c_idx * repeats + r_idx                   # [BC,1,1,RB]
    oc = c_rep // factor                              # [BC,1,1,RB]
    rem = c_rep - oc * factor                         # [BC,1,1,RB]

    # rem -> (t_i, s1, s2)
    fs2 = factor_s * factor_s
    t_i = rem // fs2                                  # [BC,1,1,RB]
    rem2 = rem - t_i * fs2
    s1 = rem2 // factor_s
    s2 = rem2 - s1 * factor_s

    # output coords
    t_g = t_in * factor_t + t_i
    t_out = t_g - t_offset                            # [BC,1,1,RB]

    h_out = h_idx * factor_s + s1                     # [BC,BH,1,RB] (broadcast h_idx over BC)
    w_out = w_idx * factor_s + s2                     # [BC,1,BW,RB] (broadcast w_idx over BC)

    # final mask: [BC,BH,BW,RB]
    m = (
        x_mask                                           # [BC,BH,BW,1] -> broadcast
        & r_ok                                           # [1,1,1,RB]
        & (oc < C_out)
        & (t_out >= 0) & (t_out < T_out)
        & (h_out < H_out)
        & (w_out < W_out)
    )

    # load h and add (broadcast x_val along R dim)
    h_off = (
        b * shB
        + oc * shC
        + t_out * shT
        + h_out * shH
        + w_out * shW
    )
    h_val = tl.load(h_ptr + h_off, mask=m, other=0)

    out_val = h_val + x_val  # x_val broadcast to [BC,BH,BW,RB]

    out_off = (
        b * soB
        + oc * soC
        + t_out * soT
        + h_out * soH
        + w_out * soW
    )
    # ✅ 一次性写回（无 for loop）
    tl.store(out_ptr + out_off, out_val, mask=m)


def dupup_fused_add(
    x: torch.Tensor,
    h: torch.Tensor,
    out_channels: int,
    factor_t: int,
    factor_s: int = 1,
    first_chunk: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    out = h + DupUp3D(x)  (fused)
    x: [B, C_in, T, H, W]
    h/out: [B, C_out, T_out, H_out, W_out]
    """
    assert x.is_cuda and h.is_cuda
    assert x.ndim == 5 and h.ndim == 5
    B, C_in, T_in, H_in, W_in = x.shape
    C_out = int(out_channels)

    factor = int(factor_t) * int(factor_s) * int(factor_s)
    assert (C_out * factor) % C_in == 0
    repeats = (C_out * factor) // C_in

    t_offset = (factor_t - 1) if first_chunk else 0
    T_out = T_in * factor_t - t_offset
    H_out = H_in * factor_s
    W_out = W_in * factor_s

    assert tuple(h.shape) == (B, C_out, T_out, H_out, W_out), \
        f"h.shape must be {(B, C_out, T_out, H_out, W_out)}, got {tuple(h.shape)}"
    assert h.dtype == x.dtype

    if out is None:
        out = torch.empty_like(h)
    else:
        assert out.is_cuda and out.shape == h.shape and out.dtype == h.dtype

    sxB, sxC, sxT, sxH, sxW = x.stride()
    shB, shC, shT, shH, shW = h.stride()
    soB, soC, soT, soH, soW = out.stride()

    # 3D grid depends on meta (BLOCK_* / R_BLOCK)
    def grid(meta):
        BC = meta["BLOCK_C"]
        BH = meta["BLOCK_H"]
        BW = meta["BLOCK_W"]
        RB = meta["R_BLOCK"]
        C_tiles = (C_in + BC - 1) // BC
        H_tiles = (H_in + BH - 1) // BH
        W_tiles = (W_in + BW - 1) // BW
        R_tiles = (repeats + RB - 1) // RB
        return (B * T_in * C_tiles * H_tiles, W_tiles, R_tiles)

    _dupup_fused_add_vecR_kernel[grid](
        x, h, out,
        B=B,
        C_in=C_in, T_in=T_in, H_in=H_in, W_in=W_in,
        C_out=C_out,
        factor_t=factor_t, factor_s=factor_s,
        repeats=repeats,
        t_offset=t_offset,
        T_out=T_out, H_out=H_out, W_out=W_out,
        sxB=sxB, sxC=sxC, sxT=sxT, sxH=sxH, sxW=sxW,
        shB=shB, shC=shC, shT=shT, shH=shH, shW=shW,
        soB=soB, soC=soC, soT=soT, soH=soH, soW=soW,
    )
    return out
