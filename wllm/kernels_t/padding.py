import torch
import triton
from wllm.kernels_t.autotune import configs_for_platform
import triton.language as tl

@triton.autotune(
    configs=configs_for_platform([
        triton.Config({"ROWS_PER_PROG": 8}, num_warps=4, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 8}, num_warps=4, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 16}, num_warps=4, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 16}, num_warps=4, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 16}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 16}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 32}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 32}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 32}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 32}, num_warps=8, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 32}, num_warps=16, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 32}, num_warps=16, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 32}, num_warps=16, num_stages=1),
        triton.Config({"ROWS_PER_PROG": 32}, num_warps=16, num_stages=1),
    ]),
    key=["n_rows", "W2"],
    cache_results=True
)
@triton.jit
def pad_lastdim_lr1_multirow_kernel(
    x_ptr, y_ptr,
    n_rows: tl.constexpr,     # total rows = prod(all dims except last)
    W: tl.constexpr,          # original width
    W2: tl.constexpr,         # W + 2
    STRIDE_ROW_X: tl.constexpr,  # elements between rows in x (contiguous => W)
    STRIDE_ROW_Y: tl.constexpr,  # elements between rows in y (contiguous => W2)
    BLOCK_W2: tl.constexpr,      # cols per row handled by one program (>= W2)
    ROWS_PER_PROG: tl.constexpr, # how many rows per program
):
    pid = tl.program_id(0)  # program id over row-blocks

    # rows this program is responsible for
    row_ids = pid * ROWS_PER_PROG + tl.arange(0, ROWS_PER_PROG)      # [R]
    row_mask = row_ids < n_rows                                       # [R]

    # columns this program covers (width dimension)
    offs_w2 = tl.arange(0, BLOCK_W2)                                  # [C]
    col_mask = offs_w2 < W2                                           # [C]

    # output col -> input col (pad_left=1)
    xw = offs_w2 - 1                                                  # [C]
    in_col_mask = (xw >= 0) & (xw < W)                                # [C]

    # Broadcast to 2D: [R, C]
    # x offsets: row_ids[:,None]*STRIDE_ROW_X + xw[None,:]
    # y offsets: row_ids[:,None]*STRIDE_ROW_Y + offs_w2[None,:]
    x_offsets = row_ids[:, None] * STRIDE_ROW_X + xw[None, :]
    y_offsets = row_ids[:, None] * STRIDE_ROW_Y + offs_w2[None, :]

    # combine masks
    x_mask = row_mask[:, None] & col_mask[None, :] & in_col_mask[None, :]
    y_mask = row_mask[:, None] & col_mask[None, :]

    # load: invalid -> 0
    x_val = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
    tl.store(y_ptr + y_offsets, x_val, mask=y_mask)


def pad_lastdim_lr1(x: torch.Tensor, rows_per_prog: int = 8) -> torch.Tensor:
    """
    x: [B,C,H,W] or [B,C,T,H,W], dtype=bfloat16
    returns y with last dim padded by (1,1), zero pad
    One Triton program handles multiple rows.
    """
    assert x.is_cuda, "x must be CUDA tensor"
    assert x.dtype == torch.bfloat16, "x must be torch.bfloat16"
    assert x.ndim in (4, 5), "only supports 4D or 5D"
    assert rows_per_prog in (1, 2, 4, 8, 16), "pick a sane ROWS_PER_PROG (1/2/4/8/16)"
    assert x.is_contiguous()
    

    W = x.shape[-1]
    W2 = W + 2
    y = torch.empty((*x.shape[:-1], W2), device=x.device, dtype=x.dtype)

    n_rows = x.numel() // W

    # packed rows assumption after contiguous
    stride_row_x = W
    stride_row_y = W2

    # Choose a block for width
    BLOCK_W2 = triton.next_power_of_2(W2)
    

    # grid over row-blocks

    grid = lambda meta: (triton.cdiv(n_rows, meta["ROWS_PER_PROG"]),)

    pad_lastdim_lr1_multirow_kernel[grid](
        x, y,
        n_rows=n_rows,
        W=W,
        W2=W2,
        STRIDE_ROW_X=stride_row_x,
        STRIDE_ROW_Y=stride_row_y,
        BLOCK_W2=BLOCK_W2
    )
    return y


# ---------------- quick test ----------------
if __name__ == "__main__":
    torch.manual_seed(0)
    x4 = torch.randn(2, 3, 4, 5, device="cuda", dtype=torch.bfloat16)
    y4 = pad_lastdim_lr1(x4, rows_per_prog=8)
    ref4 = torch.nn.functional.pad(x4, (1, 1))
    print("4D max diff:", (y4.float() - ref4.float()).abs().max().item())

    x5 = torch.randn(2, 3, 2, 4, 5, device="cuda", dtype=torch.bfloat16)
    y5 = pad_lastdim_lr1(x5, rows_per_prog=8)
    # pad only last dim => (1,1) is enough for torch F.pad with N-dim (pads last dims first)
    ref5 = torch.nn.functional.pad(x5, (1, 1))
    print("5D max diff:", (y5.float() - ref5.float()).abs().max().item())
