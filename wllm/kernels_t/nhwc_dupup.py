import triton
from wllm.kernels_t.autotune import configs_for_platform
import triton.language as tl
import torch


@triton.autotune(
    configs=configs_for_platform([
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 512}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=2),
    ]),
    key=["D", "H", "W", "Cin0", "Cout"],
    cache_results=True
)
@triton.jit
def repeat_shuffle_ndhwc_add_kernel(
    x_ptr, y_ptr, yadd_ptr,
    # sizes
    N: tl.constexpr, D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    Cin0: tl.constexpr,
    Cout: tl.constexpr,
    ft: tl.constexpr, fs: tl.constexpr,
    repeats: tl.constexpr,      # 1/2/4/8
    first_chunk: tl.constexpr,  # bool
    do_add: tl.constexpr,       # bool
    # output sizes
    Dout: tl.constexpr, Hout: tl.constexpr, Wout: tl.constexpr,
    # strides (elements) NDHWC
    xN: tl.constexpr, xD: tl.constexpr, xH: tl.constexpr, xW: tl.constexpr, xC: tl.constexpr,
    yN: tl.constexpr, yD: tl.constexpr, yH: tl.constexpr, yW: tl.constexpr, yC: tl.constexpr,
    aN: tl.constexpr, aD: tl.constexpr, aH: tl.constexpr, aW: tl.constexpr, aC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    total = N * Dout * Hout * Wout * Cout
    mask = offs < total

    # unravel output linear index -> (n, d2, h2, w2, co)
    co = offs % Cout
    t = offs // Cout
    w2 = t % Wout
    t = t // Wout
    h2 = t % Hout
    t = t // Hout
    d2 = t % Dout
    n  = t // Dout

    # apply first_chunk slicing on expanded D
    d_full = d2 + (ft - 1 if first_chunk else 0)

    # map expanded coords back to input coords
    d_in = d_full // ft
    td   = d_full - d_in * ft          # % ft

    h_in = h2 // fs
    sh   = h2 - h_in * fs              # % fs

    w_in = w2 // fs
    sw   = w2 - w_in * fs              # % fs

    # k in [0, ft*fs*fs)
    k = ((td * fs + sh) * fs + sw)
    block = ft * fs * fs

    # logical channel after repeat_interleave
    cin_logical = co * block + k
    cin0 = cin_logical // repeats

    in_bounds = mask & (d_in < D) & (h_in < H) & (w_in < W) & (cin0 < Cin0)

    x_off = n * xN + d_in * xD + h_in * xH + w_in * xW + cin0 * xC
    x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0).to(tl.float32)

    if do_add:
        a_off = n * aN + d2 * aD + h2 * aH + w2 * aW + co * aC
        a_val = tl.load(yadd_ptr + a_off, mask=mask, other=0.0).to(tl.float32)
        out = x_val + a_val
    else:
        out = x_val

    y_off = n * yN + d2 * yD + h2 * yH + w2 * yW + co * yC
    # 存回与 y dtype 一致更灵活；这里用 fp32->y dtype（bf16/fp16）由 y 决定
    tl.store(y_ptr + y_off, out.to(tl.bfloat16), mask=mask)


def fused_repeat_shuffle_ndhwc_plus_y(
    x: torch.Tensor,
    y_add: torch.Tensor,   # same output shape
    *,
    ft: int,
    fs: int,
    out_channels: int,
    repeats: int = 1,          # 1/2/4/8
    first_chunk: bool = False,
) -> torch.Tensor:
    """
    Fuse:
      out = repeat+shuffle(+first_chunk)(x) + y_add
    Input x:    [N,D,H,W,Cin0] NDHWC
    Input y_add:[N,Dout,Hout,Wout,Cout] NDHWC
    Output:     [N,Dout,Hout,Wout,Cout] bf16
    """
    if x.dim() != 5:
        raise ValueError(f"Expected x [N,D,H,W,C], got {tuple(x.shape)}")
    if y_add.dim() != 5:
        raise ValueError(f"Expected y_add 5D, got {tuple(y_add.shape)}")
    if repeats not in (1, 2, 4, 8):
        raise ValueError(f"repeats must be 1/2/4/8, got {repeats}")
    if not x.is_cuda or not y_add.is_cuda:
        raise ValueError("x and y_add must be CUDA")

    x_ = x.contiguous()
    y_add_ = y_add.contiguous()

    N, D, H, W, Cin0 = x_.shape
    ft = int(ft); fs = int(fs); Cout = int(out_channels)

    block = ft * fs * fs
    Cin = Cin0 * repeats
    if Cin % block != 0:
        raise ValueError(f"Cin(after repeat)={Cin} must be divisible by block={block}")
    if Cin // block != Cout:
        raise ValueError(f"Cin/block={Cin//block} must equal out_channels={Cout}")

    Dout_full = D * ft
    Dout = Dout_full - (ft - 1) if first_chunk else Dout_full
    Hout = H * fs
    Wout = W * fs

    if tuple(y_add_.shape) != (N, Dout, Hout, Wout, Cout):
        raise ValueError(f"y_add shape must be {(N, Dout, Hout, Wout, Cout)}, got {tuple(y_add_.shape)}")

    # 输出：bf16（和 kernel store 一致）
    y = torch.empty((N, Dout, Hout, Wout, Cout), device=x_.device, dtype=torch.bfloat16)

    xN, xD, xH, xW, xC = x_.stride()
    yN, yD, yH, yW, yC = y.stride()
    aN, aD, aH, aW, aC = y_add_.stride()

    grid = lambda meta: (triton.cdiv(N * Dout * Hout * Wout * Cout, meta["BLOCK"]),)

    repeat_shuffle_ndhwc_add_kernel[grid](
        x_, y, y_add_,
        N=N, D=D, H=H, W=W,
        Cin0=Cin0, Cout=Cout,
        ft=ft, fs=fs,
        repeats=repeats,
        first_chunk=first_chunk,
        do_add=True,
        Dout=Dout, Hout=Hout, Wout=Wout,
        xN=xN, xD=xD, xH=xH, xW=xW, xC=xC,
        yN=yN, yD=yD, yH=yH, yW=yW, yC=yC,
        aN=aN, aD=aD, aH=aH, aW=aW, aC=aC,
    )
    return y

