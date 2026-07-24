# rmsnorm_triton_bchw_bcthw.py
import torch
import triton
from wllm.kernels_t.autotune import configs_for_platform
import triton.language as tl



#_TRITON_ALLOC_CACHE = []

def triton_torch_allocator(size, alignment, stream):
    # size/alignment 单位：bytes
    # 直接分配 uint8 buffer，让 Triton 自己用它的 data_ptr()
    buf = torch.empty((size,), device="cuda", dtype=torch.uint8)
    #_TRITON_ALLOC_CACHE.append(buf)  # 防止被GC
    return buf  # <-- 只返回一个对象：Tensor（有 data_ptr）

triton.set_allocator(triton_torch_allocator)

# -------------------------
# Aggressive autotune configs (B200-ish)
# BS: spatial grouping size (B_HW or B_THW)
# -------------------------
_RMSNORM_CONFIGS = [
    # --------------------
    # Small configs: small S / small C
    # --------------------
    # triton.Config({"BLOCK_C": 64,  "BS": 8},   num_warps=4, num_stages=2),
    # triton.Config({"BLOCK_C": 128, "BS": 8},   num_warps=4, num_stages=2),

    # triton.Config({"BLOCK_C": 64,  "BS": 16},  num_warps=4, num_stages=2),
    # triton.Config({"BLOCK_C": 128, "BS": 16},  num_warps=4, num_stages=2),
    # triton.Config({"BLOCK_C": 256, "BS": 16},  num_warps=8, num_stages=3),

    # Mid: good general-purpose when S moderate
    #triton.Config({"BLOCK_C": 64,  "BS": 32},  num_warps=4, num_stages=3),
    #triton.Config({"BLOCK_C": 128, "BS": 32},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_C": 256, "BS": 32},  num_warps=8, num_stages=4),

    # --------------------
    # Aggressive configs: B200-ish throughput
    # --------------------
    # BS=32 (aggressive)
    # triton.Config({"BLOCK_C": 128, "BS": 32},  num_warps=8,  num_stages=4),
    # triton.Config({"BLOCK_C": 256, "BS": 32},  num_warps=8,  num_stages=4),
    # triton.Config({"BLOCK_C": 512, "BS": 32},  num_warps=16, num_stages=5),

    # # BS=64
    # triton.Config({"BLOCK_C": 128, "BS": 64},  num_warps=8,  num_stages=4),
    # triton.Config({"BLOCK_C": 256, "BS": 64},  num_warps=16, num_stages=5),
    # triton.Config({"BLOCK_C": 512, "BS": 64},  num_warps=16, num_stages=5),

    # # BS=128
    # triton.Config({"BLOCK_C": 128, "BS": 128}, num_warps=16, num_stages=5),
    # triton.Config({"BLOCK_C": 256, "BS": 128}, num_warps=16, num_stages=5),
]


# ============================================================
# Kernel: BCHW (x: [B,C,H,W]) but we treat HW as linear S=H*W
# ============================================================
@triton.autotune(
    configs=configs_for_platform(_RMSNORM_CONFIGS),
    key=["C", "S"],   # S = H*W
)
@triton.jit
def rmsnorm_bchw_bs_fwd_kernel(
    x_ptr, y_ptr,
    gamma_ptr, bias_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    S: tl.constexpr,  # S = H*W
    sb: tl.constexpr, sc: tl.constexpr, sh: tl.constexpr, sw: tl.constexpr,
    yb: tl.constexpr, yc: tl.constexpr, yh: tl.constexpr, yw: tl.constexpr,
    sg0: tl.constexpr, sb0: tl.constexpr,
    eps: tl.constexpr,
    has_bias: tl.constexpr,
    BS: tl.constexpr,        # autotuned
    BLOCK_C: tl.constexpr,   # autotuned
):
    pid = tl.program_id(0)  # over (b, s_block)
    s_blocks = (S + BS - 1) // BS
    b = pid // s_blocks
    sbk = pid - b * s_blocks

    s0 = sbk * BS
    offs_s = s0 + tl.arange(0, BS)              # [BS]
    mask_s = offs_s < S

    # decode s -> (h, w)
    h = offs_s // W
    w = offs_s - h * W

    # offsets within sample b for each s: h*sh + w*sw
    off_spatial = h * sh + w * sw               # [BS]

    # base pointers for each s in [BS]
    x_base = x_ptr + b * sb + off_spatial       # [BS]
    y_base = y_ptr + b * yb + off_spatial       # [BS]

    # ---- Pass 1: sumsq over C for each s ----
    sumsq = tl.zeros((BS,), dtype=tl.float32)
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)     # [BC]
        mask_c = offs_c < C

        x = tl.load(
            x_base[None, :] + offs_c[:, None] * sc,
            mask=mask_c[:, None] & mask_s[None, :],
            other=0.0
        ).to(tl.float32)

        sumsq += tl.sum(x * x, axis=0)

    denom = tl.rsqrt(sumsq + eps)                       # [BS]
    scale = tl.sqrt(tl.full((BS,), C, tl.float32))      # [BS]

    # ---- Pass 2: write output ----
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        x = tl.load(
            x_base[None, :] + offs_c[:, None] * sc,
            mask=mask_c[:, None] & mask_s[None, :],
            other=0.0
        ).to(tl.float32)

        g = tl.load(gamma_ptr + offs_c * sg0, mask=mask_c, other=1.0).to(tl.float32)  # [BC]
        y = x * denom[None, :] * scale[None, :] * g[:, None]

        if has_bias:
            bv = tl.load(bias_ptr + offs_c * sb0, mask=mask_c, other=0.0).to(tl.float32)
            y += bv[:, None]

        tl.store(
            y_base[None, :] + offs_c[:, None] * yc,
            y.to(tl.bfloat16),
            mask=mask_c[:, None] & mask_s[None, :]
        )


def rmsnorm_bchw(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    x:     [B, C, H, W]  CUDA
    gamma: (C,1,1)       CUDA
    bias:  (C,1,1) or None
    output: bfloat16
    """
    assert x.is_cuda and x.dim() == 4, f"x must be CUDA [B,C,H,W], got {tuple(x.shape)}"
    B, C, H, W = x.shape
    S = H * W

    assert gamma.is_cuda and gamma.shape == (C, 1, 1), f"gamma must be (C,1,1), got {tuple(gamma.shape)}"
    if bias is not None:
        assert bias.is_cuda and bias.shape == (C, 1, 1), f"bias must be (C,1,1), got {tuple(bias.shape)}"

    x_ = x.contiguous()
    y = torch.empty((B, C, H, W), device=x.device, dtype=torch.bfloat16)

    sb, sc, sh, sw = x_.stride()
    yb, yc, yh, yw = y.stride()

    sg0 = gamma.stride(0)
    sb0 = bias.stride(0) if bias is not None else sg0

    grid = lambda meta: (B * triton.cdiv(S, meta["BS"]),)

    rmsnorm_bchw_bs_fwd_kernel[grid](
        x_, y,
        gamma, bias if bias is not None else gamma,
        B=B, C=C, H=H, W=W,
        S=S,
        sb=sb, sc=sc, sh=sh, sw=sw,
        yb=yb, yc=yc, yh=yh, yw=yw,
        sg0=sg0, sb0=sb0,
        eps=eps,
        has_bias=(bias is not None),
    )
    return y


# ============================================================
# Kernel: BCTHW (x: [B,C,T,H,W]) treat THW as linear S=T*H*W
# ============================================================
@triton.autotune(
    configs=configs_for_platform(_RMSNORM_CONFIGS),
    key=["C", "S"],   # S = T*H*W
)
@triton.jit
def rmsnorm_bcthw_bs_fwd_kernel(
    x_ptr, y_ptr,
    gamma_ptr, bias_ptr,
    B: tl.constexpr, C: tl.constexpr, T: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    S: tl.constexpr,  # S = T*H*W
    sb: tl.constexpr, sc: tl.constexpr, st: tl.constexpr, sh: tl.constexpr, sw: tl.constexpr,
    yb: tl.constexpr, yc: tl.constexpr, yt: tl.constexpr, yh: tl.constexpr, yw: tl.constexpr,
    sg0: tl.constexpr, sb0: tl.constexpr,
    eps: tl.constexpr,
    has_bias: tl.constexpr,
    BS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)  # over (b, s_block)
    s_blocks = (S + BS - 1) // BS
    b = pid // s_blocks
    sbk = pid - b * s_blocks

    s0 = sbk * BS
    offs_s = s0 + tl.arange(0, BS)              # [BS]
    mask_s = offs_s < S

    hw = H * W
    t = offs_s // hw
    r = offs_s - t * hw
    h = r // W
    w = r - h * W

    off_spatial = t * st + h * sh + w * sw      # [BS]

    x_base = x_ptr + b * sb + off_spatial       # [BS]
    y_base = y_ptr + b * yb + off_spatial       # [BS]

    # ---- Pass 1: sumsq over C for each s ----
    sumsq = tl.zeros((BS,), dtype=tl.float32)
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        x = tl.load(
            x_base[None, :] + offs_c[:, None] * sc,
            mask=mask_c[:, None] & mask_s[None, :],
            other=0.0
        ).to(tl.float32)

        sumsq += tl.sum(x * x, axis=0)

    denom = tl.rsqrt(sumsq + eps)
    scale = tl.sqrt(tl.full((BS,), C, tl.float32))

    # ---- Pass 2: write output ----
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        x = tl.load(
            x_base[None, :] + offs_c[:, None] * sc,
            mask=mask_c[:, None] & mask_s[None, :],
            other=0.0
        ).to(tl.float32)

        g = tl.load(gamma_ptr + offs_c * sg0, mask=mask_c, other=1.0).to(tl.float32)
        y = x * denom[None, :] * scale[None, :] * g[:, None]

        if has_bias:
            bv = tl.load(bias_ptr + offs_c * sb0, mask=mask_c, other=0.0).to(tl.float32)
            y += bv[:, None]

        tl.store(
            y_base[None, :] + offs_c[:, None] * yc,
            y.to(tl.bfloat16),
            mask=mask_c[:, None] & mask_s[None, :]
        )


def rmsnorm_bcthw(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    x:     [B, C, T, H, W]  CUDA
    gamma: (C,1,1,1)        CUDA
    bias:  (C,1,1,1) or None
    output: bfloat16
    """
    assert x.is_cuda and x.dim() == 5, f"x must be CUDA [B,C,T,H,W], got {tuple(x.shape)}"
    B, C, T, H, W = x.shape
    S = T * H * W

    assert gamma.is_cuda and gamma.shape == (C, 1, 1, 1), f"gamma must be (C,1,1,1), got {tuple(gamma.shape)}"
    if bias is not None:
        assert bias.is_cuda and bias.shape == (C, 1, 1, 1), f"bias must be (C,1,1,1), got {tuple(bias.shape)}"

    x_ = x.contiguous()
    y = torch.empty((B, C, T, H, W), device=x.device, dtype=torch.bfloat16)

    sb, sc, st, sh, sw = x_.stride()
    yb, yc, yt, yh, yw = y.stride()

    sg0 = gamma.stride(0)
    sb0 = bias.stride(0) if bias is not None else sg0

    grid = lambda meta: (B * triton.cdiv(S, meta["BS"]),)

    rmsnorm_bcthw_bs_fwd_kernel[grid](
        x_, y,
        gamma, bias if bias is not None else gamma,
        B=B, C=C, T=T, H=H, W=W,
        S=S,
        sb=sb, sc=sc, st=st, sh=sh, sw=sw,
        yb=yb, yc=yc, yt=yt, yh=yh, yw=yw,
        sg0=sg0, sb0=sb0,
        eps=eps,
        has_bias=(bias is not None),
    )
    return y



# rmsnorm_tma.py
import torch
import triton
import triton.language as tl

# 更全一点：加小 config + 激进 config（B200）
_RMSNORM_TMA_CONFIGS = [
    # small
    triton.Config({"BLOCK_C": 64,  "BS": 8},   num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_C": 128, "BS": 8},   num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_C": 64,  "BS": 16},  num_warps=4,  num_stages=2),
    triton.Config({"BLOCK_C": 128, "BS": 16},  num_warps=4,  num_stages=2),

    # mid
    triton.Config({"BLOCK_C": 128, "BS": 32},  num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_C": 256, "BS": 32},  num_warps=8,  num_stages=4),

    # aggressive (B200-ish)
    triton.Config({"BLOCK_C": 128, "BS": 64},  num_warps=8,  num_stages=4),
    triton.Config({"BLOCK_C": 256, "BS": 64},  num_warps=16, num_stages=5),
    triton.Config({"BLOCK_C": 512, "BS": 64},  num_warps=16, num_stages=5),
    triton.Config({"BLOCK_C": 256, "BS": 128}, num_warps=16, num_stages=5),
]


@triton.autotune(
    configs=configs_for_platform(_RMSNORM_TMA_CONFIGS),
    key=["C", "S"],
)
@triton.jit
def rmsnorm_bcs_tma_kernel(
    x_ptr, y_ptr,
    gamma_ptr, bias_ptr,
    B: tl.constexpr, C: tl.constexpr, S: tl.constexpr,
    sb: tl.constexpr, sc: tl.constexpr, ss: tl.constexpr,   # ss should be 1 for contiguous S
    yb: tl.constexpr, yc: tl.constexpr, ys: tl.constexpr,
    sg0: tl.constexpr, sb0: tl.constexpr,                    # gamma/bias stride on dim0
    eps: tl.constexpr,
    has_bias: tl.constexpr,
    BS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)  # over (b, s_block)
    s_blocks = tl.cdiv(S, BS)
    b = pid // s_blocks
    sbk = pid - b * s_blocks
    s0 = sbk * BS

    offs_s = s0 + tl.arange(0, BS)  # [BS]
    mask_s = offs_s < S

    # base pointer for this batch
    x_b = x_ptr + b * sb
    y_b = y_ptr + b * yb

    # --- Create TMA descriptors for x and y as 2D tensors [C, S] ---
    # NOTE: TMA constraints:
    #  - last dim must be contiguous (stride 1)
    #  - leading strides must be 16-byte aligned (bf16 => stride multiple of 8)
    #  - only 2-5D supported
    # See docs.
    desc_x = tl.make_tensor_descriptor(
        x_b,
        shape=[C, S],
        strides=[sc, ss],                 # ss should be 1
        block_shape=[BLOCK_C, BS],
    )
    desc_y = tl.make_tensor_descriptor(
        y_b,
        shape=[C, S],
        strides=[yc, ys],                 # ys should be 1
        block_shape=[BLOCK_C, BS],
    )

    # ---- Pass 1: sumsq over C for each s in the block ----
    sumsq = tl.zeros((BS,), dtype=tl.float32)

    # We sweep C by BLOCK_C tiles
    for c0 in range(0, C, BLOCK_C):
        # TMA descriptor load a tile [BLOCK_C, BS] starting at offsets [c0, s0]
        x_tile = tl.load_tensor_descriptor(desc_x, [c0, s0]).to(tl.float32)
        # mask out tail S if needed (tile includes full BS)
        x_tile = tl.where(mask_s[None, :], x_tile, 0.0)
        sumsq += tl.sum(x_tile * x_tile, axis=0)

    denom = tl.rsqrt(sumsq + eps)
    scale = tl.sqrt(tl.full((BS,), C, tl.float32))

    # ---- Pass 2: write output (TMA store) ----
    for c0 in range(0, C, BLOCK_C):
        x_tile = tl.load_tensor_descriptor(desc_x, [c0, s0]).to(tl.float32)
        x_tile = tl.where(mask_s[None, :], x_tile, 0.0)

        offs_c = c0 + tl.arange(0, BLOCK_C)        # [BLOCK_C]
        mask_c = offs_c < C

        g = tl.load(gamma_ptr + offs_c * sg0, mask=mask_c, other=1.0).to(tl.float32)[:, None]
        y_tile = x_tile * denom[None, :] * scale[None, :] * g

        if has_bias:
            bv = tl.load(bias_ptr + offs_c * sb0, mask=mask_c, other=0.0).to(tl.float32)[:, None]
            y_tile += bv

        # mask out invalid C rows + tail S cols
        mask_2d = mask_c[:, None] & mask_s[None, :]
        y_tile = tl.where(mask_2d, y_tile, 0.0)

        # TMA store: tile shape must match desc_y.block_shape
        tl.store_tensor_descriptor(desc_y, [c0, s0], y_tile.to(tl.bfloat16))


def _check_tma_alignment_for_bf16(stride_c: int) -> None:
    # TMA doc: leading strides must be multiples of 16-byte strides.
    # bf16 is 2 bytes => stride_c must be multiple of 8.
    if stride_c % 8 != 0:
        raise ValueError(
            f"TMA requires leading stride multiple of 16 bytes. For bf16, stride_c must be multiple of 8, got {stride_c}."
        )


def rmsnorm_bchw_tma(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    x: [B,C,H,W], gamma/bias: (C,1,1). output bf16.
    """
    if getattr(torch.version, "hip", None) is not None or torch.cuda.get_device_capability()[0] < 9:
        return rmsnorm_bchw(x, gamma, bias, eps)
    assert x.is_cuda and x.dim() == 4
    B, C, H, W = x.shape
    S = H * W

    assert gamma.is_cuda and gamma.shape == (C, 1, 1)
    if bias is not None:
        assert bias.is_cuda and bias.shape == (C, 1, 1)

    # reshape to [B,C,S] with contiguous S
    x_ = x.contiguous().view(B, C, S)
    y = torch.empty((B, C, S), device=x.device, dtype=torch.bfloat16)

    sb, sc, ss = x_.stride()  # expect ss == 1
    yb, yc, ys = y.stride()   # expect ys == 1
    if ss != 1 or ys != 1:
        raise ValueError("Expected contiguous S dimension (stride == 1) for TMA descriptor.")

    _check_tma_alignment_for_bf16(sc)
    _check_tma_alignment_for_bf16(yc)

    sg0 = gamma.stride(0)
    sb0 = bias.stride(0) if bias is not None else sg0

    grid = lambda meta: (B * triton.cdiv(S, meta["BS"]),)

    rmsnorm_bcs_tma_kernel[grid](
        x_, y,
        gamma, bias if bias is not None else gamma,
        B=B, C=C, S=S,
        sb=sb, sc=sc, ss=ss,
        yb=yb, yc=yc, ys=ys,
        sg0=sg0, sb0=sb0,
        eps=eps,
        has_bias=(bias is not None),
    )

    return y.view(B, C, H, W)


def rmsnorm_bcthw_tma(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    x: [B,C,T,H,W], gamma/bias: (C,1,1,1). output bf16.
    """
    if getattr(torch.version, "hip", None) is not None or torch.cuda.get_device_capability()[0] < 9:
        return rmsnorm_bcthw(x, gamma, bias, eps)
    assert x.is_cuda and x.dim() == 5
    B, C, T, H, W = x.shape
    S = T * H * W

    assert gamma.is_cuda and gamma.shape == (C, 1, 1, 1)
    if bias is not None:
        assert bias.is_cuda and bias.shape == (C, 1, 1, 1)

    x_ = x.contiguous().view(B, C, S)
    y = torch.empty((B, C, S), device=x.device, dtype=torch.bfloat16)

    sb, sc, ss = x_.stride()
    yb, yc, ys = y.stride()
    if ss != 1 or ys != 1:
        raise ValueError("Expected contiguous S dimension (stride == 1) for TMA descriptor.")

    _check_tma_alignment_for_bf16(sc)
    _check_tma_alignment_for_bf16(yc)

    sg0 = gamma.stride(0)
    sb0 = bias.stride(0) if bias is not None else sg0

    grid = lambda meta: (B * triton.cdiv(S, meta["BS"]),)

    rmsnorm_bcs_tma_kernel[grid](
        x_, y,
        gamma, bias if bias is not None else gamma,
        B=B, C=C, S=S,
        sb=sb, sc=sc, ss=ss,
        yb=yb, yc=yc, ys=ys,
        sg0=sg0, sb0=sb0,
        eps=eps,
        has_bias=(bias is not None),
    )

    return y.view(B, C, T, H, W)



@triton.autotune(
    configs=configs_for_platform(_RMSNORM_CONFIGS),
    key=["C", "S"],
)
@triton.jit
def rmsnorm_bcs_silu_fwd_kernel(
    x_ptr, y_ptr,
    gamma_ptr, bias_ptr,
    B: tl.constexpr, C: tl.constexpr, S: tl.constexpr,
    sb: tl.constexpr, sc: tl.constexpr, ss: tl.constexpr,   # x strides (elements)
    yb: tl.constexpr, yc: tl.constexpr, ys: tl.constexpr,   # y strides (elements)
    sg0: tl.constexpr, sb0: tl.constexpr,                   # gamma/bias stride on dim0
    eps: tl.constexpr,
    has_bias: tl.constexpr,
    BS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # program over (b, s_block)
    pid = tl.program_id(0)
    s_blocks = tl.cdiv(S, BS)
    b = pid // s_blocks
    sbk = pid - b * s_blocks

    s0 = sbk * BS
    offs_s = s0 + tl.arange(0, BS)       # [BS]
    mask_s = offs_s < S

    # base pointers for this batch and s-tile
    x_base = x_ptr + b * sb + offs_s * ss    # [BS]
    y_base = y_ptr + b * yb + offs_s * ys    # [BS]

    # ---- Pass 1: sumsq over C for each s ----
    sumsq = tl.zeros((BS,), dtype=tl.float32)

    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)  # [BC]
        mask_c = offs_c < C

        x = tl.load(
            x_base[None, :] + offs_c[:, None] * sc,
            mask=mask_c[:, None] & mask_s[None, :],
            other=0.0
        ).to(tl.float32)

        sumsq += tl.sum(x * x, axis=0)       # [BS]

    denom = tl.rsqrt(sumsq + eps)                  # [BS]
    scale = tl.sqrt(tl.full((BS,), C, tl.float32)) # [BS]  sqrt(C)

    # ---- Pass 2: affine + SiLU + store ----
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        x = tl.load(
            x_base[None, :] + offs_c[:, None] * sc,
            mask=mask_c[:, None] & mask_s[None, :],
            other=0.0
        ).to(tl.float32)

        g = tl.load(gamma_ptr + offs_c * sg0, mask=mask_c, other=1.0).to(tl.float32)  # [BC]
        y = x * denom[None, :] * scale[None, :] * g[:, None]

        if has_bias:
            bv = tl.load(bias_ptr + offs_c * sb0, mask=mask_c, other=0.0).to(tl.float32)
            y += bv[:, None]

        # SiLU: y * sigmoid(y)
        y = y * tl.sigmoid(y)

        tl.store(
            y_base[None, :] + offs_c[:, None] * yc,
            y.to(tl.bfloat16),
            mask=mask_c[:, None] & mask_s[None, :]
        )


def rmsnorm_bchw_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    x:     [B, C, H, W] CUDA
    gamma: (C, 1, 1) CUDA
    bias:  (C, 1, 1) or None
    output: bfloat16, includes SiLU
    """
    assert x.is_cuda and x.dim() == 4, f"x must be CUDA [B,C,H,W], got {tuple(x.shape)}"
    B, C, H, W = x.shape
    S = H * W

    assert gamma.is_cuda and gamma.shape == (C, 1, 1), f"gamma must be (C,1,1), got {tuple(gamma.shape)}"
    if bias is not None:
        assert bias.is_cuda and bias.shape == (C, 1, 1), f"bias must be (C,1,1), got {tuple(bias.shape)}"

    # flatten spatial to S, keep S contiguous
    x_ = x.contiguous().view(B, C, S)
    y_ = torch.empty((B, C, S), device=x.device, dtype=torch.bfloat16)

    sb, sc, ss = x_.stride()   # typically (C*S, S, 1)
    yb, yc, ys = y_.stride()   # typically (C*S, S, 1)

    sg0 = gamma.stride(0)
    sb0 = bias.stride(0) if bias is not None else sg0

    grid = lambda meta: (B * triton.cdiv(S, meta["BS"]),)

    rmsnorm_bcs_silu_fwd_kernel[grid](
        x_, y_,
        gamma, bias if bias is not None else gamma,
        B=B, C=C, S=S,
        sb=sb, sc=sc, ss=ss,
        yb=yb, yc=yc, ys=ys,
        sg0=sg0, sb0=sb0,
        eps=eps,
        has_bias=(bias is not None),
    )
    return y_.view(B, C, H, W)


def rmsnorm_bcthw_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    x:     [B, C, T, H, W] CUDA
    gamma: (C, 1, 1, 1) CUDA
    bias:  (C, 1, 1, 1) or None
    output: bfloat16, includes SiLU
    """
    assert x.is_cuda and x.dim() == 5, f"x must be CUDA [B,C,T,H,W], got {tuple(x.shape)}"
    B, C, T, H, W = x.shape
    S = T * H * W

    assert gamma.is_cuda and gamma.shape == (C, 1, 1, 1), f"gamma must be (C,1,1,1), got {tuple(gamma.shape)}"
    if bias is not None:
        assert bias.is_cuda and bias.shape == (C, 1, 1, 1), f"bias must be (C,1,1,1), got {tuple(bias.shape)}"

    x_ = x.contiguous().view(B, C, S)
    y_ = torch.empty((B, C, S), device=x.device, dtype=torch.bfloat16)

    sb, sc, ss = x_.stride()
    yb, yc, ys = y_.stride()

    sg0 = gamma.stride(0)
    sb0 = bias.stride(0) if bias is not None else sg0

    grid = lambda meta: (B * triton.cdiv(S, meta["BS"]),)

    rmsnorm_bcs_silu_fwd_kernel[grid](
        x_, y_,
        gamma, bias if bias is not None else gamma,
        B=B, C=C, S=S,
        sb=sb, sc=sc, ss=ss,
        yb=yb, yc=yc, ys=ys,
        sg0=sg0, sb0=sb0,
        eps=eps,
        has_bias=(bias is not None),
    )
    return y_.view(B, C, T, H, W)

