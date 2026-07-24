import triton
from wllm.kernels_t.autotune import configs_for_platform
import triton.language as tl
import torch

_RMSNORM_NDHWC_CONFIGS = [
    # 常规稳妥
    triton.Config({"BS": 128, "BLOCK_C": 128}, num_warps=4,  num_stages=4),
    triton.Config({"BS": 128, "BLOCK_C": 256}, num_warps=8,  num_stages=4),
    triton.Config({"BS": 64,  "BLOCK_C": 256}, num_warps=4,  num_stages=4),
    triton.Config({"BS": 16,  "BLOCK_C": 256}, num_warps=8,  num_stages=2),
    triton.Config({"BS": 16,  "BLOCK_C": 128}, num_warps=8,  num_stages=2),
    triton.Config({"BS": 16,  "BLOCK_C": 128}, num_warps=4,  num_stages=2),
    triton.Config({"BS": 8,  "BLOCK_C": 256}, num_warps=4,  num_stages=2),
    triton.Config({"BS": 8,  "BLOCK_C": 128}, num_warps=4,  num_stages=2),
    triton.Config({"BS": 4,  "BLOCK_C": 256}, num_warps=4,  num_stages=2),
    triton.Config({"BS": 4,  "BLOCK_C": 128}, num_warps=4,  num_stages=2),
    triton.Config({"BS": 4,  "BLOCK_C": 64}, num_warps=4,  num_stages=2),

    # B200 倾向：更大 BS / 更大 BLOCK_C
    triton.Config({"BS": 256, "BLOCK_C": 256}, num_warps=8,  num_stages=5),
    triton.Config({"BS": 256, "BLOCK_C": 512}, num_warps=8,  num_stages=6),
    triton.Config({"BS": 128, "BLOCK_C": 512}, num_warps=8,  num_stages=6),

    # C 很大时可能更强（但寄存器压力更大，autotune 会淘汰不合适的）
    triton.Config({"BS": 256, "BLOCK_C": 1024}, num_warps=16, num_stages=6),
    triton.Config({"BS": 128, "BLOCK_C": 1024}, num_warps=16, num_stages=5),
]


@triton.autotune(
    configs=configs_for_platform(_RMSNORM_NDHWC_CONFIGS),
    key=["C", "S"],
    cache_results=True
)
@triton.jit
def rmsnorm_bsc_mul_silu_fwd_kernel(
    x_ptr, y_ptr,
    gamma_ptr,
    B: tl.constexpr, S: tl.constexpr, C: tl.constexpr,
    xb: tl.constexpr, xs: tl.constexpr, xc: tl.constexpr,   # x strides (elements) for [B,S,C]
    yb: tl.constexpr, ys: tl.constexpr, yc: tl.constexpr,   # y strides (elements) for [B,S,C]
    sg: tl.constexpr,                                       # gamma stride
    eps: tl.constexpr,
    BS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # program over (b, s_block)
    pid = tl.program_id(0)
    s_blocks = tl.cdiv(S, BS)
    b = pid // s_blocks
    sbk = pid - b * s_blocks

    s0 = sbk * BS
    offs_s = s0 + tl.arange(0, BS)     # [BS]
    mask_s = offs_s < S

    # pointers to x[b, offs_s, 0] / y[b, offs_s, 0]
    x_base = x_ptr + b * xb + offs_s * xs
    y_base = y_ptr + b * yb + offs_s * ys

    # ---- Pass 1: sumsq over C for each s ----
    sumsq = tl.zeros((BS,), dtype=tl.float32)

    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)   # [BC]
        mask_c = offs_c < C

        # x: [BS, BC]
        x = tl.load(
            x_base[:, None] + offs_c[None, :] * xc,
            mask=mask_s[:, None] & mask_c[None, :],
            other=0.0,
        ).to(tl.float32)

        sumsq += tl.sum(x * x, axis=1)        # [BS]

    denom = tl.rsqrt(sumsq + eps)                  # [BS]
    scale = tl.sqrt(tl.full((BS,), C, tl.float32)) # [BS] sqrt(C)

    # ---- Pass 2: norm + mul + silu + store ----
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        x = tl.load(
            x_base[:, None] + offs_c[None, :] * xc,
            mask=mask_s[:, None] & mask_c[None, :],
            other=0.0,
        ).to(tl.float32)

        g = tl.load(gamma_ptr + offs_c * sg, mask=mask_c, other=1.0).to(tl.float32)  # [BC]

        y = x * denom[:, None] * scale[:, None] * g[None, :]

        # SiLU
        y = y * tl.sigmoid(y)

        tl.store(
            y_base[:, None] + offs_c[None, :] * yc,
            y.to(tl.bfloat16),
            mask=mask_s[:, None] & mask_c[None, :],
        )


def rmsnorm_ndhwc_mul_silu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    x:     [B, D, H, W, C] CUDA (NDHWC)
    gamma: [C] or anything reshaped to [C]
    output: [B, D, H, W, C] bfloat16, includes SiLU
    """
    assert x.is_cuda and x.dim() == 5, f"x must be CUDA [B,D,H,W,C], got {tuple(x.shape)}"
    B, D, H, W, C = x.shape
    S = D * H * W

    assert gamma.is_cuda, "gamma must be CUDA"
    gamma_ = gamma.reshape(-1)
    assert gamma_.numel() == C, f"gamma must have C={C} elements, got {gamma_.numel()}"

    # 确保 C 连续（NDHWC 常见是连续的；如果不连续，这里会拷贝一次）
    x_ = x.contiguous()

    # 你想要的输出：直接分配 NDHWC
    y_ = torch.empty((B, D, H, W, C), device=x.device, dtype=torch.bfloat16)

    # 仅用于 kernel 寻址的临时 view（返回时不 view）
    x3 = x_.view(B, S, C)
    y3 = y_.view(B, S, C)

    xb, xs, xc = x3.stride()
    yb, ys, yc = y3.stride()
    sg = gamma_.stride(0)

    grid = lambda meta: (B * triton.cdiv(S, meta["BS"]),)

    rmsnorm_bsc_mul_silu_fwd_kernel[grid](
        x3, y3, gamma_,
        B=B, S=S, C=C,
        xb=xb, xs=xs, xc=xc,
        yb=yb, ys=ys, yc=yc,
        sg=sg,
        eps=eps,
    )
    return y_


_L2NORM_CONFIGS = [
    triton.Config({"BS": 128, "BLOCK_C": 128}, num_warps=4),
    triton.Config({"BS": 128, "BLOCK_C": 256}, num_warps=8),
    triton.Config({"BS": 64,  "BLOCK_C": 128}, num_warps=4),
    triton.Config({"BS": 64,  "BLOCK_C": 256}, num_warps=8),
]


@triton.autotune(configs=configs_for_platform(_L2NORM_CONFIGS), key=["C", "S"], cache_results=True)
@triton.jit
def l2norm_bias_mul_silu_fwd_kernel(
    x_ptr, y_ptr,
    gamma_ptr, bias_ptr,
    B: tl.constexpr, S: tl.constexpr, C: tl.constexpr,
    xb: tl.constexpr, xs: tl.constexpr, xc: tl.constexpr,   # [B,S,C]
    yb: tl.constexpr, ys: tl.constexpr, yc: tl.constexpr,   # [B,S,C]
    sg: tl.constexpr, sb0: tl.constexpr,                    # gamma/bias stride
    scale: tl.constexpr,                                     # scalar
    eps: tl.constexpr,
    BS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    s_blocks = tl.cdiv(S, BS)
    b = pid // s_blocks
    sbk = pid - b * s_blocks

    s0 = sbk * BS
    offs_s = s0 + tl.arange(0, BS)              # [BS]
    mask_s = offs_s < S

    # base pointers for x[b, offs_s, 0] / y[b, offs_s, 0]
    x_base = x_ptr + b * xb + offs_s * xs
    y_base = y_ptr + b * yb + offs_s * ys

    # Pass1: sum (x+bias)^2 over C for each s
    sumsq = tl.zeros((BS,), dtype=tl.float32)

    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)     # [BC]
        mask_c = offs_c < C

        x = tl.load(
            x_base[:, None] + offs_c[None, :] * xc,
            mask=mask_s[:, None] & mask_c[None, :],
            other=0.0
        ).to(tl.float32)

        b0 = tl.load(bias_ptr + offs_c * sb0, mask=mask_c, other=0.0).to(tl.float32)  # [BC]
        xb0 = x + b0[None, :]

        sumsq += tl.sum(xb0 * xb0, axis=1)

    inv_norm = tl.rsqrt(sumsq + eps)            # [BS]

    # Pass2: normalize + scale*gamma + SiLU + store
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        x = tl.load(
            x_base[:, None] + offs_c[None, :] * xc,
            mask=mask_s[:, None] & mask_c[None, :],
            other=0.0
        ).to(tl.float32)

        b0 = tl.load(bias_ptr + offs_c * sb0, mask=mask_c, other=0.0).to(tl.float32)
        g  = tl.load(gamma_ptr + offs_c * sg,  mask=mask_c, other=1.0).to(tl.float32)

        y = (x + b0[None, :]) * inv_norm[:, None] * scale * g[None, :]

        y = y * tl.sigmoid(y)  # SiLU

        tl.store(
            y_base[:, None] + offs_c[None, :] * yc,
            y.to(tl.bfloat16),
            mask=mask_s[:, None] & mask_c[None, :]
        )


def l2norm_bias_mul_silu_ndhwc_5d(
    x: torch.Tensor,          # [B,D,H,W,C]
    gamma: torch.Tensor,      # [C] or reshapeable to [C]
    bias: torch.Tensor,       # [C] or reshapeable to [C]
    scale: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    y = SiLU( normalize(x + bias, dim=-1) * scale * gamma )
    Input/Output are 5D NDHWC.
    Output dtype: bfloat16
    """
    if x.dim() != 5:
        raise ValueError(f"Expected input [B,D,H,W,C], got {tuple(x.shape)}")
    if not x.is_cuda:
        raise ValueError("x must be CUDA")

    x_ = x.contiguous()
    B, D, H, W, C = x_.shape
    S = D * H * W

    gamma_ = gamma.reshape(-1).contiguous()
    bias_  = bias.reshape(-1).contiguous()
    if gamma_.numel() != C or bias_.numel() != C:
        raise ValueError(f"gamma/bias must have C={C} elements; got gamma={gamma_.numel()} bias={bias_.numel()}")

    # 直接分配 5D 输出（你要求的）
    y = torch.empty((B, D, H, W, C), device=x_.device, dtype=torch.bfloat16)

    # 仅作为 kernel 视图，不作为返回值
    x3 = x_.view(B, S, C)
    y3 = y.view(B, S, C)

    xb, xs, xc = x3.stride()
    yb, ys, yc = y3.stride()

    grid = lambda meta: (B * triton.cdiv(S, meta["BS"]),)

    l2norm_bias_mul_silu_fwd_kernel[grid](
        x3, y3,
        gamma_, bias_,
        B=B, S=S, C=C,
        xb=xb, xs=xs, xc=xc,
        yb=yb, ys=ys, yc=yc,
        sg=gamma_.stride(0),
        sb0=bias_.stride(0),
        scale=scale,
        eps=eps,
    )
    return y




@triton.autotune(
    configs=configs_for_platform(_RMSNORM_NDHWC_CONFIGS),
    key=["C", "S"],
    cache_results=True
)
@triton.jit
def rmsnorm_bsc_mul_fwd_kernel(
    x_ptr, y_ptr,
    gamma_ptr,
    B: tl.constexpr, S: tl.constexpr, C: tl.constexpr,
    xb: tl.constexpr, xs: tl.constexpr, xc: tl.constexpr,   # x strides (elements) for [B,S,C]
    yb: tl.constexpr, ys: tl.constexpr, yc: tl.constexpr,   # y strides (elements) for [B,S,C]
    sg: tl.constexpr,                                       # gamma stride
    eps: tl.constexpr,
    BS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # program over (b, s_block)
    pid = tl.program_id(0)
    s_blocks = tl.cdiv(S, BS)
    b = pid // s_blocks
    sbk = pid - b * s_blocks

    s0 = sbk * BS
    offs_s = s0 + tl.arange(0, BS)     # [BS]
    mask_s = offs_s < S

    # pointers to x[b, offs_s, 0] / y[b, offs_s, 0]
    x_base = x_ptr + b * xb + offs_s * xs
    y_base = y_ptr + b * yb + offs_s * ys

    # ---- Pass 1: sumsq over C for each s ----
    sumsq = tl.zeros((BS,), dtype=tl.float32)

    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)   # [BC]
        mask_c = offs_c < C

        # x: [BS, BC]
        x = tl.load(
            x_base[:, None] + offs_c[None, :] * xc,
            mask=mask_s[:, None] & mask_c[None, :],
            other=0.0,
        ).to(tl.float32)

        sumsq += tl.sum(x * x, axis=1)        # [BS]

    denom = tl.rsqrt(sumsq + eps)                  # [BS]
    scale = tl.sqrt(tl.full((BS,), C, tl.float32)) # [BS] sqrt(C)

    # ---- Pass 2: norm + mul + silu + store ----
    for c0 in range(0, C, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        mask_c = offs_c < C

        x = tl.load(
            x_base[:, None] + offs_c[None, :] * xc,
            mask=mask_s[:, None] & mask_c[None, :],
            other=0.0,
        ).to(tl.float32)

        g = tl.load(gamma_ptr + offs_c * sg, mask=mask_c, other=1.0).to(tl.float32)  # [BC]

        y = x * denom[:, None] * scale[:, None] * g[None, :]

        tl.store(
            y_base[:, None] + offs_c[None, :] * yc,
            y.to(tl.bfloat16),
            mask=mask_s[:, None] & mask_c[None, :],
        )


@torch.compile(disable=True)
def rmsnorm_ndhwc_mul(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    x:     [B, D, H, W, C] CUDA (NDHWC)
    gamma: [C] or anything reshaped to [C]
    output: [B, D, H, W, C] bfloat16, includes SiLU
    """
    assert x.is_cuda and x.dim() == 5, f"x must be CUDA [B,D,H,W,C], got {tuple(x.shape)}"
    B, D, H, W, C = x.shape
    S = D * H * W

    assert gamma.is_cuda, "gamma must be CUDA"
    gamma_ = gamma.reshape(-1)
    assert gamma_.numel() == C, f"gamma must have C={C} elements, got {gamma_.numel()}"

    # 确保 C 连续（NDHWC 常见是连续的；如果不连续，这里会拷贝一次）
    x_ = x.contiguous()

    # 你想要的输出：直接分配 NDHWC
    y_ = torch.empty((B, D, H, W, C), device=x.device, dtype=torch.bfloat16)

    # 仅用于 kernel 寻址的临时 view（返回时不 view）
    x3 = x_.view(B, S, C)
    y3 = y_.view(B, S, C)

    xb, xs, xc = x3.stride()
    yb, ys, yc = y3.stride()
    sg = gamma_.stride(0)

    grid = lambda meta: (B * triton.cdiv(S, meta["BS"]),)

    rmsnorm_bsc_mul_fwd_kernel[grid](
        x3, y3, gamma_,
        B=B, S=S, C=C,
        xb=xb, xs=xs, xc=xc,
        yb=yb, ys=ys, yc=yc,
        sg=sg,
        eps=eps,
    )
    return y_