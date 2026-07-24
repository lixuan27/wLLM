import torch
import triton
import triton.language as tl
from typing import List
@triton.jit
def fuse_pad_ndhwc_kernel(
    x_ptr, y_ptr,
    # n_rows 是最终输出的 rows: B*C*T_out*H_out
    n_rows,                      # runtime scalar
    W: tl.constexpr,             # original W
    W2: tl.constexpr,            # FINAL output width = (W+2) + pwL + pwR
    STRIDE_ROW_X: tl.constexpr,  # x row stride (contiguous => W)
    STRIDE_ROW_Y: tl.constexpr,  # y row stride (contiguous => W2)
    BLOCK_W2: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,

    # shapes
    B: tl.constexpr, C: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    T_OUT: tl.constexpr, H_OUT: tl.constexpr,

    # padding_list for F.pad(h, [pwL,pwR, phL,phR, ptL,ptR])
    PW_L: tl.constexpr, PW_R: tl.constexpr,
    PH_L: tl.constexpr, PH_R: tl.constexpr,
    PT_L: tl.constexpr, PT_R: tl.constexpr,
):
    
    pass


def fuse_pad_ndhwc(
x: torch.Tensor,
padding_list: List[int]
):
    
    assert x.is_contiguous(torch.channels_last_3d)

    B, C, T, H, W = x.shape
    pwL, pwR, phL, phR, ptL, ptR = [int(v) for v in padding_list]

    # first pad_lastdim_lr1 gives W+2, then F.pad adds pwL+pwR
    W2 = (W + 2) + pwL + pwR
    T_OUT = T + ptL + ptR
    H_OUT = H + phL + phR
    
    y = torch.empty((B, C, T_OUT, H_OUT, W2), device=x.device, dtype=x.dtype)



