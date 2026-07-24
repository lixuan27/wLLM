import torch
import triton
from wllm.kernels_t.autotune import configs_for_platform
import triton.language as tl

@triton.autotune(
    configs=configs_for_platform([
        triton.Config({"BLK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLK": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLK": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLK": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLK": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLK": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLK": 1024}, num_warps=8, num_stages=3),
    ]),
    key=["D"],
    cache_results=True
)
@triton.jit
def norm_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    eps,
    D: tl.constexpr,
    BLK: tl.constexpr = 512
):
   
    pid = tl.program_id(0)

    x_row = x_ptr + pid * D
    o_row = o_ptr + pid * D
    w_row = w_ptr
    b_row = b_ptr

    offs = tl.arange(0, BLK)

    # hints
    tl.multiple_of(x_row, 16)
    tl.multiple_of(o_row, 16)
    tl.multiple_of(w_row, 16)
    tl.multiple_of(b_row, 16)
    tl.max_contiguous(offs, 128)

    # pass 1
    sum_ = tl.zeros((), dtype=tl.float32)
    sumsq = tl.zeros((), dtype=tl.float32)

    for base in tl.static_range(0, D, BLK):
        x = tl.load(x_row + base + offs, mask=base+offs<D, other=0.0).to(tl.float32)
        sum_ += tl.sum(x, axis=0)
        sumsq += tl.sum(x * x, axis=0)

    mean = sum_ / D
    var = sumsq / D - mean * mean
    rstd = tl.rsqrt(var + eps)

    # pass 2
    for base in tl.static_range(0, D, BLK):
        x = tl.load(x_row + base + offs, mask=base+offs<D).to(tl.float32)
        w = tl.load(w_row + base + offs, mask=base+offs<D).to(tl.float32)
        b = tl.load(b_row + base + offs, mask=base+offs<D).to(tl.float32)

        y = (x - mean) * (rstd * w) + b
        tl.store(o_row + base + offs, y.to(tl.bfloat16), mask=base+offs<D)

@torch.compile(disable=True)
def norm_fp32(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    B, S, D = x.shape
    assert B == 1
    assert weight.shape == (D,)
    assert bias.shape == (D,)

    o = torch.empty((B, S, D), device=x.device, dtype=x.dtype)
    grid = (S,)
    norm_kernel[grid](x, weight, bias, o, eps, D)
    return o



@triton.autotune(
    configs=configs_for_platform([
        triton.Config({"BLK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLK": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLK": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLK": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLK": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLK": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLK": 1024}, num_warps=8, num_stages=3),
    ]),
    key=["D"],
    cache_results=True
)
@triton.jit
def norm_scale_shift_kernel(
    x_ptr,
    scale_msa_ptr,
    shift_msa_ptr,
    o_ptr,
    token_length,
    eps,
    D: tl.constexpr,
    BLK: tl.constexpr = 512
):
    
    pid = tl.program_id(0)
    msa_id = pid // token_length

    x_row = x_ptr + pid * D
    o_row = o_ptr + pid * D
    scale_msa_row = scale_msa_ptr + msa_id * D
    shift_msa_row = shift_msa_ptr + msa_id * D


    offs = tl.arange(0, BLK)

    # ---------- pass 1: mean / var ----------
    sum_ = tl.zeros((), dtype=tl.float32)
    sumsq = tl.zeros((), dtype=tl.float32)

    tl.multiple_of(x_row, 16)

    for base in tl.static_range(0, D, BLK):
        x = tl.load(x_row + base + offs, mask=base+offs<D).to(tl.float32)
        sum_ += tl.sum(x, axis=0)
        sumsq += tl.sum(x * x, axis=0)

    mean = sum_ / D
    var = sumsq / D - mean * mean
    rstd = tl.rsqrt(var + eps)

    # ---------- pass 2: normalize ----------
    for base in tl.static_range(0, D, BLK):
        x = tl.load(x_row + base + offs, mask=base+offs<D, other=0.0).to(tl.float32)
        scale = tl.load(scale_msa_row + base + offs, mask=base+offs<D).to(tl.float32)
        shift = tl.load(shift_msa_row + base + offs, mask=base+offs<D).to(tl.float32)

        y = (((x - mean) * rstd) * (1 + scale) + shift).to(tl.bfloat16)
        tl.store(o_row + base + offs, y, mask=base+offs<D)


@torch.compile(disable=True)
def norm_scale_shift(
    x: torch.Tensor,
    scale_msa: torch.Tensor,
    shift_msa: torch.Tensor,
    token_length: int,
    eps: float
):  
    
    B, S, D = x.shape
    assert B == 1
    assert scale_msa.is_contiguous()
    assert shift_msa.is_contiguous()
    assert S % token_length == 0
    assert x.is_contiguous()
    o = torch.empty_like(x)
    grid = (S,)
    norm_scale_shift_kernel[grid](x, scale_msa, shift_msa, o, token_length, eps, D)

    return o



@triton.autotune(
    configs=configs_for_platform([
        triton.Config({"BLK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLK": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLK": 512}, num_warps=4, num_stages=3),
        triton.Config({"BLK": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLK": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLK": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLK": 1024}, num_warps=8, num_stages=3),
    ]),
    key=["D"],
    cache_results=True
)
@triton.jit
def scale_residual_kernel(
    x_ptr,
    y_ptr,
    gate_msa_ptr,
    o_ptr,
    token_length,
    D: tl.constexpr,
    BLK: tl.constexpr = 512
):
    pid = tl.program_id(0)
    msa_id = pid // token_length

    x_row = x_ptr + pid * D
    y_row = y_ptr + pid * D
    o_row = o_ptr + pid * D
    gate_msa_row = gate_msa_ptr + msa_id * D
    offs = tl.arange(0, BLK)


    # ---------- pass 2: normalize ----------
    for base in tl.static_range(0, D, BLK):
        mask = base + offs < D
        x = tl.load(x_row + base + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_row + base + offs, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_msa_row + base + offs, mask=mask, other=0.0).to(tl.float32)
        z = (x + y * gate).to(tl.bfloat16)
        tl.store(o_row + base + offs, z, mask=mask)


@torch.compile(disable=True)
def scale_residual(
    x: torch.Tensor,
    y: torch.Tensor,
    gate_msa: torch.Tensor,
    token_length: int,
):  
    
    B, S, D = x.shape
    assert B == 1
    assert y.shape == (B, S, D)
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert gate_msa.is_contiguous()
    assert S % token_length == 0
    o = torch.empty_like(x)
    grid = (S,)

    scale_residual_kernel[grid](x, y, gate_msa, o, token_length, D)

    return o
