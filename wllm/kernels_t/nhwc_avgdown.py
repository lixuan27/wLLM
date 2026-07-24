import torch
import triton
import triton.language as tl

@triton.jit
def causal_patch_groupmean_add_y_nthwc_kernel(
    x_ptr, y_ptr, out_ptr,
    T: tl.constexpr, C: tl.constexpr,
    T2: tl.constexpr, H2: tl.constexpr, W2: tl.constexpr,
    OUT_CH: tl.constexpr, GROUP: tl.constexpr,
    PAD_T: tl.constexpr,
    stride_xb: tl.constexpr, stride_xt: tl.constexpr, stride_xh: tl.constexpr, stride_xw: tl.constexpr, stride_xc: tl.constexpr,
    stride_yb: tl.constexpr, stride_yt: tl.constexpr, stride_yh: tl.constexpr, stride_yw: tl.constexpr, stride_yc: tl.constexpr,
    stride_ob: tl.constexpr, stride_ot: tl.constexpr, stride_oh: tl.constexpr, stride_ow: tl.constexpr, stride_oc: tl.constexpr,
    FT: tl.constexpr, FS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    """
    x:   (B, T,  H,  W,  C)      NTHWC
    y:   (B, T2, H2, W2, OUT_CH) NTHWOC
    out: (B, T2, H2, W2, OUT_CH) NTHWOC
    out = mean_group(patchify(x)) + y
    """

    pid = tl.program_id(axis=0)

    oc_blocks = tl.cdiv(OUT_CH, BLOCK_OC)
    pix = pid // oc_blocks
    oc_block = pid - pix * oc_blocks

    # decode pix -> (b,t2,h2,w2), order: (((b*T2+t2)*H2+h2)*W2+w2)
    w2 = pix % W2
    tmp = pix // W2
    h2 = tmp % H2
    tmp = tmp // H2
    t2 = tmp % T2
    b  = tmp // T2  # valid because grid is exact (see host)

    oc = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc < OUT_CH

    # fp32 accumulator for forward(x)
    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    factor = FT * FS * FS

    # forward(x) reduction over GROUP
    for g in tl.static_range(0, GROUP):
        k = oc * GROUP + g

        c = k // factor
        r = k - c * factor

        dt = r // (FS * FS)
        rr = r - dt * (FS * FS)
        dh = rr // FS
        dw = rr - dh * FS

        # causal left pad: t = t2*FT + dt - PAD_T
        t = t2 * FT + dt - PAD_T
        h = h2 * FS + dh
        w = w2 * FS + dw

        t_in = (t >= 0) & (t < T)
        c_in = c < C

        x_off = (
            b * stride_xb +
            t * stride_xt +
            h * stride_xh +
            w * stride_xw +
            c * stride_xc
        )
        m = oc_mask & t_in & c_in
        x_val = tl.load(x_ptr + x_off, mask=m, other=0.0).to(tl.float32)
        acc += x_val

    acc = acc / GROUP  # fp32

    # load y and add
    y_off = (
        b * stride_yb +
        t2 * stride_yt +
        h2 * stride_yh +
        w2 * stride_yw +
        oc * stride_yc
    )
    y_val = tl.load(y_ptr + y_off, mask=oc_mask, other=0.0).to(tl.float32)

    out_val = acc + y_val

    # store (casts to out tensor dtype = x.dtype)
    o_off = (
        b * stride_ob +
        t2 * stride_ot +
        h2 * stride_oh +
        w2 * stride_ow +
        oc * stride_oc
    )
    tl.store(out_ptr + o_off, out_val, mask=oc_mask)


def fused_forward_add_y_triton_nthwc(
    x: torch.Tensor,
    y: torch.Tensor,
    factor_t: int,
    factor_s: int,
    out_channels: int,
) -> torch.Tensor:
    """
    Returns out = forward(x) + y, fused in one Triton kernel.

    x: (B,T,H,W,C) CUDA, dtype fp16/bf16/fp32
    y: (B,T2,H2,W2,out_channels) CUDA, same dtype as x
    out: same shape/dtype as y (and x.dtype)
    """
    assert x.is_cuda and y.is_cuda, "x and y must be CUDA tensors"
    assert x.ndim == 5, "x must be (B,T,H,W,C)"
    assert y.ndim == 5, "y must be (B,T2,H2,W2,out_channels)"
    assert x.dtype == y.dtype, "for fused add, require x.dtype == y.dtype"
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32), f"unsupported dtype: {x.dtype}"

    B, T, H, W, C = x.shape
    FT, FS = factor_t, factor_s
    assert H % FS == 0 and W % FS == 0, "H and W must be divisible by factor_s"

    factor = FT * FS * FS
    assert (C * factor) % out_channels == 0, "C*factor must be divisible by out_channels"
    group_size = (C * factor) // out_channels

    # causal-left-pad amount (conceptual); handled via masking
    Tpad = ((T + FT - 1) // FT) * FT
    pad_t = Tpad - T

    T2 = Tpad // FT
    H2 = H // FS
    W2 = W // FS

    assert y.shape == (B, T2, H2, W2, out_channels), \
        f"y shape must be (B,T2,H2,W2,out_channels) = {(B,T2,H2,W2,out_channels)}, got {tuple(y.shape)}"

    out = torch.empty_like(y)  # ✅ dtype same as x, shape same as y

    sx = x.stride()
    sy = y.stride()
    so = out.stride()

    BLOCK_OC = 64 if out_channels >= 64 else 32
    oc_blocks = triton.cdiv(out_channels, BLOCK_OC)

    # exact grid: one program per (pix, oc_block)
    grid = (B * T2 * H2 * W2 * oc_blocks,)

    causal_patch_groupmean_add_y_nthwc_kernel[grid](
        x, y, out,
        T=T, C=C,
        T2=T2, H2=H2, W2=W2,
        OUT_CH=out_channels, GROUP=group_size,
        PAD_T=pad_t,
        stride_xb=sx[0], stride_xt=sx[1], stride_xh=sx[2], stride_xw=sx[3], stride_xc=sx[4],
        stride_yb=sy[0], stride_yt=sy[1], stride_yh=sy[2], stride_yw=sy[3], stride_yc=sy[4],
        stride_ob=so[0], stride_ot=so[1], stride_oh=so[2], stride_ow=so[3], stride_oc=so[4],
        FT=FT, FS=FS,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
    )
    return out