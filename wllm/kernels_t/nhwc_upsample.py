#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NDHWC nearest-exact 3D upsample (gather) via Triton, with B200-leaning autotune.

Assumptions:
- Input is NDHWC contiguous (standard contiguous layout in PyTorch for 5D NDHWC tensor):
  strides == (D*H*W*C, H*W*C, W*C, C, 1)
- Output is NDHWC contiguous with outD/outH/outW = int(in*scale) unless you override.

What this provides:
- Triton kernel: upsample_nearest_exact_ndhwc_kernel
- Autotune configs (B200-leaning): B200_UPSAMPLE_AUTOTUNE_CONFIGS
- Wrapper: upsample_nearest_exact_ndhwc(x, scale_d, scale_h, scale_w)
- Optional correctness test vs torch.nn.functional.interpolate (NCDHW reference)
- Optional micro-benchmark

Run:
  python upsample_ndhwc_nearest_exact_triton.py
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Tuple

import torch
import triton
from wllm.kernels_t.autotune import configs_for_platform
import triton.language as tl


# -----------------------------
# B200-leaning autotune configs
# -----------------------------
B200_UPSAMPLE_AUTOTUNE_CONFIGS = [
    # Small C / general
    triton.Config({"BLOCK_S": 8,  "BLOCK_C": 64},  num_warps=4,  num_stages=3),
    triton.Config({"BLOCK_S": 16, "BLOCK_C": 64},  num_warps=4,  num_stages=3),
    triton.Config({"BLOCK_S": 8,  "BLOCK_C": 128}, num_warps=4,  num_stages=3),
    triton.Config({"BLOCK_S": 16, "BLOCK_C": 128}, num_warps=8,  num_stages=3),

    # Mid C (common)
    triton.Config({"BLOCK_S": 8,  "BLOCK_C": 256}, num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_S": 16, "BLOCK_C": 256}, num_warps=8,  num_stages=3),
    triton.Config({"BLOCK_S": 32, "BLOCK_C": 128}, num_warps=8,  num_stages=3),

    # Large C
    triton.Config({"BLOCK_S": 8,  "BLOCK_C": 512}, num_warps=8,  num_stages=4),
    triton.Config({"BLOCK_S": 16, "BLOCK_C": 512}, num_warps=16, num_stages=4),
    triton.Config({"BLOCK_S": 32, "BLOCK_C": 256}, num_warps=16, num_stages=4),

    # Tiny spatial fallback
    triton.Config({"BLOCK_S": 4,  "BLOCK_C": 64},  num_warps=2,  num_stages=2),
    triton.Config({"BLOCK_S": 4,  "BLOCK_C": 128}, num_warps=2,  num_stages=2),
]


def _autotune():
    # Keyed on "overall spatial work" and channels.
    return triton.autotune(
        configs=configs_for_platform(B200_UPSAMPLE_AUTOTUNE_CONFIGS),
        key=["out_total_spatial", "C"],
        cache_results=True
    )


# -----------------------------
# Kernel
# -----------------------------
@_autotune()
@triton.jit
def upsample_nearest_exact_ndhwc_kernel(
    X_ptr, Y_ptr,
    # runtime scalars
    out_total_spatial: tl.constexpr,
    # shapes
    N: tl.constexpr,
    inD: tl.constexpr, inH: tl.constexpr, inW: tl.constexpr,
    outD: tl.constexpr, outH: tl.constexpr, outW: tl.constexpr,
    C: tl.constexpr,
    # scales (float)
    sd: tl.constexpr, sh: tl.constexpr, sw: tl.constexpr,
    # tiling
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    2D grid:
      pid_s -> blocks over linearized (N*outD*outH*outW)
      pid_c -> blocks over C
    """
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)

    s_offsets = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)     # [BS]
    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)     # [BC]

    s_mask = s_offsets < out_total_spatial
    c_mask = c_offsets < C

    out_spatial = outD * outH * outW

    # Decode linear index -> (n, od, oh, ow)
    n = s_offsets // out_spatial
    rem = s_offsets - n * out_spatial
    od = rem // (outH * outW)
    rem2 = rem - od * (outH * outW)
    oh = rem2 // outW
    ow = rem2 - oh * outW

    # nearest-exact (center-aligned):
    # in_idx = floor((out + 0.5) / scale)
    # (equivalent to round((out+0.5)/scale - 0.5) with standard round-half-up behavior)
    od_f = od.to(tl.float32) + 0.5
    oh_f = oh.to(tl.float32) + 0.5
    ow_f = ow.to(tl.float32) + 0.5

    id_ = tl.floor(od_f / sd).to(tl.int32)
    ih_ = tl.floor(oh_f / sh).to(tl.int32)
    iw_ = tl.floor(ow_f / sw).to(tl.int32)

    # clamp
    id_ = tl.maximum(0, tl.minimum(id_, inD - 1))
    ih_ = tl.maximum(0, tl.minimum(ih_, inH - 1))
    iw_ = tl.maximum(0, tl.minimum(iw_, inW - 1))

    # NDHWC contiguous strides
    strideN = inD * inH * inW * C
    strideD = inH * inW * C
    strideH = inW * C
    strideW = C

    out_strideN = outD * outH * outW * C
    out_strideD = outH * outW * C
    out_strideH = outW * C
    out_strideW = C

    x_base = n * strideN + id_ * strideD + ih_ * strideH + iw_ * strideW  # [BS]
    y_base = n * out_strideN + od * out_strideD + oh * out_strideH + ow * out_strideW  # [BS]

    # pointers [BS, BC]
    x_ptrs = X_ptr + x_base[:, None] + c_offsets[None, :]
    y_ptrs = Y_ptr + y_base[:, None] + c_offsets[None, :]

    # Helpful alignment hints (safe if NDHWC contiguous and C often multiple of 16)
    tl.multiple_of(c_offsets, 16)

    mask = s_mask[:, None] & c_mask[None, :]
    vals = tl.load(x_ptrs, mask=mask, other=0.0)
    tl.store(y_ptrs, vals, mask=mask)


# -----------------------------
# Wrapper
# -----------------------------
def _assert_ndhwc_contiguous(x: torch.Tensor):
    assert x.ndim == 5, "expected 5D NDHWC"
    assert x.is_cuda, "expected CUDA tensor"
    assert x.is_contiguous(), "expected contiguous NDHWC tensor (standard contiguous layout)"
    N, D, H, W, C = x.shape
    exp = (D * H * W * C, H * W * C, W * C, C, 1)
    assert x.stride() == exp, f"expected NDHWC contiguous strides {exp}, got {x.stride()}"


def upsample_nearest_exact_ndhwc(
    x: torch.Tensor,
    scale_d: float = 2.0,
    scale_h: float = 2.0,
    scale_w: float = 2.0,
    *,
    out_sizes: Tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """
    x: NDHWC contiguous
    Returns: NDHWC contiguous
    """
    _assert_ndhwc_contiguous(x)
    assert scale_d > 0 and scale_h > 0 and scale_w > 0

    N, inD, inH, inW, C = x.shape
    if out_sizes is None:
        outD = int(inD * scale_d)
        outH = int(inH * scale_h)
        outW = int(inW * scale_w)
    else:
        outD, outH, outW = out_sizes
        scale_d = outD / inD
        scale_h = outH / inH
        scale_w = outW / inW

    y = torch.empty((N, outD, outH, outW, C), device=x.device, dtype=x.dtype)

    out_total_spatial = N * outD * outH * outW

    # Grid over (spatial blocks, channel blocks)
    # Use max BLOCK_S/C from configs? Autotune will override per config.
    # We'll pick a reasonable default for initial launch shape:
    default_block_s = 16
    default_block_c = 128

    grid = (
        triton.cdiv(out_total_spatial, default_block_s),
        triton.cdiv(C, default_block_c),
    )

    upsample_nearest_exact_ndhwc_kernel[grid](
        x, y,
        out_total_spatial=out_total_spatial,
        N=N,
        inD=inD, inH=inH, inW=inW,
        outD=outD, outH=outH, outW=outW,
        C=C,
        sd=float(scale_d), sh=float(scale_h), sw=float(scale_w),
    )
    return y


# -----------------------------
# Reference & tests
# -----------------------------
def reference_torch_interpolate_ndhwc(
    x_ndhwc: torch.Tensor,
    scale_d: float,
    scale_h: float,
    scale_w: float,
) -> torch.Tensor:
    """
    Torch reference uses NCDHW, so permute.
    Returns NDHWC.
    """
    import torch.nn.functional as F
    # NDHWC -> NCDHW
    x = x_ndhwc.permute(0, 4, 1, 2, 3).contiguous()
    y = F.interpolate(
        x,
        scale_factor=(scale_d, scale_h, scale_w),
        mode="nearest-exact",
    )
    # NCDHW -> NDHWC
    return y.permute(0, 2, 3, 4, 1).contiguous()


@dataclass
class BenchResult:
    name: str
    ms: float
    gbps: float


def _bytes_moved(N, inD, inH, inW, outD, outH, outW, C, dtype: torch.dtype) -> int:
    # Each output element reads one input element (C channels) and writes one output element.
    # Bytes: read + write
    elem = torch.tensor([], dtype=dtype).element_size()
    out_elems = N * outD * outH * outW * C
    return 2 * out_elems * elem


@torch.no_grad()
def benchmark(
    x: torch.Tensor,
    scale_d: float,
    scale_h: float,
    scale_w: float,
    iters: int = 100,
    warmup: int = 20,
) -> list[BenchResult]:
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        y = upsample_nearest_exact_ndhwc(x, scale_d, scale_h, scale_w)
    torch.cuda.synchronize()

    # Timing Triton
    t0 = time.time()
    for _ in range(iters):
        y = upsample_nearest_exact_ndhwc(x, scale_d, scale_h, scale_w)
    torch.cuda.synchronize()
    t1 = time.time()
    triton_ms = (t1 - t0) * 1000.0 / iters

    # Timing torch reference
    for _ in range(warmup):
        yr = reference_torch_interpolate_ndhwc(x, scale_d, scale_h, scale_w)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        yr = reference_torch_interpolate_ndhwc(x, scale_d, scale_h, scale_w)
    torch.cuda.synchronize()
    t1 = time.time()
    torch_ms = (t1 - t0) * 1000.0 / iters

    N, inD, inH, inW, C = x.shape
    outD = int(inD * scale_d)
    outH = int(inH * scale_h)
    outW = int(inW * scale_w)
    bytes_rw = _bytes_moved(N, inD, inH, inW, outD, outH, outW, C, x.dtype)
    gb = bytes_rw / 1e9

    return [
        BenchResult("triton", triton_ms, gb / (triton_ms / 1000.0)),
        BenchResult("torch_interpolate", torch_ms, gb / (torch_ms / 1000.0)),
    ]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = "cuda"
    torch.manual_seed(0)

    # You can tweak these to match your workload
    N, D, H, W, C = 2, 24, 64, 64, 256
    scale_d, scale_h, scale_w = 2.0, 2.0, 2.0

    # NDHWC contiguous
    x = torch.randn((N, D, H, W, C), device=device, dtype=torch.float16).contiguous()
    _assert_ndhwc_contiguous(x)

    # Correctness
    y_triton = upsample_nearest_exact_ndhwc(x, scale_d, scale_h, scale_w)
    y_ref = reference_torch_interpolate_ndhwc(x, scale_d, scale_h, scale_w)

    max_abs = (y_triton - y_ref).abs().max().item()
    same = torch.equal(y_triton, y_ref)

    print("Correctness:")
    print(f"  torch.equal: {same}")
    print(f"  max_abs_err: {max_abs}")

    # Benchmark
    print("\nBenchmark (ms, GB/s):")
    results = benchmark(x, scale_d, scale_h, scale_w, iters=100, warmup=20)
    for r in results:
        print(f"  {r.name:16s} {r.ms:8.3f} ms   {r.gbps:8.2f} GB/s")


if __name__ == "__main__":
    # Make autotune deterministic-ish across runs if you want:
    # os.environ["TRITON_AUTOTUNE_CACHE_DIR"] = "/tmp/triton_autotune_cache"
    main()
